//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FIRContext.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

#define DEBUG_TYPE "flang-codegen"

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "TypeConverter.h"

namespace {
/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::ConvertOpToLLVMPattern<FromOp> {
public:
  explicit FIROpConversion(fir::LLVMTypeConverter &lowering)
      : mlir::ConvertOpToLLVMPattern<FromOp>(lowering) {}

protected:
  mlir::Type convertType(mlir::Type ty) const {
    return lowerTy().convertType(ty);
  }

  mlir::LLVM::ConstantOp
  genConstantOffset(mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    int offset) const {
    auto ity = lowerTy().offsetType();
    auto cattr = rewriter.getI32IntegerAttr(offset);
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
  }

  /// Construct code sequence to get the rank from a box.
  mlir::Value getRankFromBox(mlir::Location loc, mlir::Value box,
                             mlir::Type resultTy,
                             mlir::ConversionPatternRewriter &rewriter) const {
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    mlir::LLVM::ConstantOp cRank =
        genConstantOffset(loc, rewriter, kRankPosInBox);
    auto pty = mlir::LLVM::LLVMPointerType::get(resultTy);
    auto p = rewriter.create<mlir::LLVM::GEPOp>(
        loc, pty, mlir::ValueRange{box, c0, cRank});
    return rewriter.create<mlir::LLVM::LoadOp>(loc, resultTy, p);
  }

  /// Method to construct code sequence to get the triple for dimension `dim`
  /// from a box.
  SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, ArrayRef<mlir::Type> retTys,
                 mlir::Value box, mlir::Value dim,
                 mlir::ConversionPatternRewriter &rewriter) const {
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    mlir::LLVM::ConstantOp cDims =
        genConstantOffset(loc, rewriter, kDimsPosInBox);
    mlir::LLVM::LoadOp l0 =
        loadFromOffset(loc, box, c0, cDims, dim, 0, retTys[0], rewriter);
    mlir::LLVM::LoadOp l1 =
        loadFromOffset(loc, box, c0, cDims, dim, 1, retTys[1], rewriter);
    mlir::LLVM::LoadOp l2 =
        loadFromOffset(loc, box, c0, cDims, dim, 2, retTys[2], rewriter);
    return {l0.getResult(), l1.getResult(), l2.getResult()};
  }

  mlir::LLVM::LoadOp
  loadFromOffset(mlir::Location loc, mlir::Value a, mlir::LLVM::ConstantOp c0,
                 mlir::LLVM::ConstantOp cDims, mlir::Value dim, int off,
                 mlir::Type ty,
                 mlir::ConversionPatternRewriter &rewriter) const {
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    mlir::LLVM::ConstantOp c = genConstantOffset(loc, rewriter, off);
    mlir::LLVM::GEPOp p = genGEP(loc, pty, rewriter, a, c0, cDims, dim, c);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  /// Read base address from a fir.box. Returned address has type ty.
  mlir::Value
  loadBaseAddrFromBox(mlir::Location loc, mlir::Type ty, mlir::Value box,
                      mlir::ConversionPatternRewriter &rewriter) const {
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    mlir::LLVM::ConstantOp cAddr =
        genConstantOffset(loc, rewriter, kAddrPosInBox);
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    mlir::LLVM::GEPOp p = genGEP(loc, pty, rewriter, box, c0, cAddr);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  mlir::Value
  loadElementSizeFromBox(mlir::Location loc, mlir::Type ty, mlir::Value box,
                         mlir::ConversionPatternRewriter &rewriter) const {
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    mlir::LLVM::ConstantOp cElemLen =
        genConstantOffset(loc, rewriter, kElemLenPosInBox);
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    mlir::LLVM::GEPOp p = genGEP(loc, pty, rewriter, box, c0, cElemLen);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }

  template <typename... ARGS>
  mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::Type ty,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value base, ARGS... args) const {
    SmallVector<mlir::Value> cv{args...};
    return rewriter.create<mlir::LLVM::GEPOp>(loc, ty, base, cv);
  }

  fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<fir::LLVMTypeConverter *>(this->getTypeConverter());
  }
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpAndTypeConversion : public FIROpConversion<FromOp> {
public:
  using FIROpConversion<FromOp>::FIROpConversion;
  using OpAdaptor = typename FromOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(FromOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type ty = this->convertType(op.getType());
    return doRewrite(op, ty, adaptor, rewriter);
  }

  virtual mlir::LogicalResult
  doRewrite(FromOp addr, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const = 0;
};

// Lower `fir.address_of` operation to `llvm.address_of` operation.
struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp addr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(addr.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
        addr, ty, addr.symbol().getRootReference().getValue());
    return success();
  }
};

/// Lower `fir.box_addr` to the sequence of operations to extract the first
/// element of the box.
struct BoxAddrOpConversion : public FIROpConversion<fir::BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxAddrOp boxaddr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxaddr.getLoc();
    mlir::Type ty = convertType(boxaddr.getType());
    if (auto argty = boxaddr.val().getType().dyn_cast<fir::BoxType>()) {
      rewriter.replaceOp(boxaddr, loadBaseAddrFromBox(loc, ty, a, rewriter));
    } else {
      auto c0attr = rewriter.getI32IntegerAttr(0);
      auto c0 = mlir::ArrayAttr::get(boxaddr.getContext(), c0attr);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxaddr, ty, a,
                                                              c0);
    }
    return success();
  }
};

/// Lower `fir.box_dims` to a sequence of operations to extract the requested
/// dimension infomartion from the boxed value.
/// Result in a triple set of GEPs and loads.
struct BoxDimsOpConversion : public FIROpConversion<fir::BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxDimsOp boxdims, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type, 3> resultTypes = {
        convertType(boxdims.getResult(0).getType()),
        convertType(boxdims.getResult(1).getType()),
        convertType(boxdims.getResult(2).getType()),
    };
    auto results =
        getDimsFromBox(boxdims.getLoc(), resultTypes, adaptor.getOperands()[0],
                       adaptor.getOperands()[1], rewriter);
    rewriter.replaceOp(boxdims, results);
    return success();
  }
};

/// Lower `fir.box_elesize` to a sequence of operations ro extract the size of
/// an element in the boxed value.
struct BoxEleSizeOpConversion : public FIROpConversion<fir::BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxEleSizeOp boxelesz, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxelesz.getLoc();
    auto ty = convertType(boxelesz.getType());
    rewriter.replaceOp(boxelesz, loadElementSizeFromBox(loc, ty, a, rewriter));
    return success();
  }
};

/// Lower `fir.box_rank` to the sequence of operation to extract the rank from
/// the box.
struct BoxRankOpConversion : public FIROpConversion<fir::BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxRankOp boxrank, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxrank.getLoc();
    mlir::Type ty = convertType(boxrank.getType());
    auto result = getRankFromBox(loc, a, ty, rewriter);
    rewriter.replaceOp(boxrank, result);
    return success();
  }
};

// `fir.call` -> `llvm.call`
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CallOp call, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        call, resultTys, adaptor.getOperands(), call->getAttrs());
    return success();
  }
};

static mlir::Type getComplexEleTy(mlir::Type complex) {
  if (auto cc = complex.dyn_cast<mlir::ComplexType>())
    return cc.getElementType();
  return complex.cast<fir::ComplexType>().getElementType();
}

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public FIROpConversion<fir::ConvertOp> {
  using FIROpConversion::FIROpConversion;

  static bool isFloatingPointTy(mlir::Type ty) {
    return ty.isa<mlir::FloatType>();
  }

  mlir::LogicalResult
  matchAndRewrite(fir::ConvertOp convert, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto fromTy = convertType(convert.value().getType());
    auto toTy = convertType(convert.res().getType());
    mlir::Value op0 = adaptor.getOperands()[0];
    if (fromTy == toTy) {
      rewriter.replaceOp(convert, op0);
      return success();
    }
    auto loc = convert.getLoc();
    auto convertFpToFp = [&](mlir::Value val, unsigned fromBits,
                             unsigned toBits, mlir::Type toTy) -> mlir::Value {
      if (fromBits == toBits) {
        // TODO: Converting between two floating-point representations with the
        // same bitwidth is not allowed for now.
        mlir::emitError(loc,
                        "cannot implicitly convert between two floating-point "
                        "representations of the same bitwidth");
        return {};
      }
      if (fromBits > toBits)
        return rewriter.create<mlir::LLVM::FPTruncOp>(loc, toTy, val);
      return rewriter.create<mlir::LLVM::FPExtOp>(loc, toTy, val);
    };
    // Complex to complex conversion.
    if (fir::isa_complex(convert.value().getType()) &&
        fir::isa_complex(convert.res().getType())) {
      // Special case: handle the conversion of a complex such that both the
      // real and imaginary parts are converted together.
      auto zero = mlir::ArrayAttr::get(convert.getContext(),
                                       rewriter.getI32IntegerAttr(0));
      auto one = mlir::ArrayAttr::get(convert.getContext(),
                                      rewriter.getI32IntegerAttr(1));
      auto ty = convertType(getComplexEleTy(convert.value().getType()));
      auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, op0, zero);
      auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, op0, one);
      auto nt = convertType(getComplexEleTy(convert.res().getType()));
      auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(ty);
      auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(nt);
      auto rc = convertFpToFp(rp, fromBits, toBits, nt);
      auto ic = convertFpToFp(ip, fromBits, toBits, nt);
      auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, toTy);
      auto i1 =
          rewriter.create<mlir::LLVM::InsertValueOp>(loc, toTy, un, rc, zero);
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(convert, toTy, i1,
                                                             ic, one);
      return mlir::success();
    }
    // Floating point to floating point conversion.
    if (isFloatingPointTy(fromTy)) {
      if (isFloatingPointTy(toTy)) {
        auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        auto v = convertFpToFp(op0, fromBits, toBits, toTy);
        rewriter.replaceOp(convert, v);
        return mlir::success();
      }
      if (toTy.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (fromTy.isa<mlir::IntegerType>()) {
      // Integer to integer conversion.
      if (toTy.isa<mlir::IntegerType>()) {
        auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        assert(fromBits != toBits);
        if (fromBits > toBits) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(convert, toTy, op0);
          return mlir::success();
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(convert, toTy, op0);
        return mlir::success();
      }
      // Integer to floating point conversion.
      if (isFloatingPointTy(toTy)) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(convert, toTy, op0);
        return mlir::success();
      }
      // Integer to pointer conversion.
      if (toTy.isa<mlir::LLVM::LLVMPointerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (fromTy.isa<mlir::LLVM::LLVMPointerType>()) {
      // Pointer to integer conversion.
      if (toTy.isa<mlir::IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(convert, toTy, op0);
        return mlir::success();
      }
      // Pointer to pointer conversion.
      if (toTy.isa<mlir::LLVM::LLVMPointerType>()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(convert, toTy, op0);
        return mlir::success();
      }
    }
    return emitError(loc) << "cannot convert " << fromTy << " to " << toTy;
  }
};

/// Lower `fir.has_value` operation to `llvm.return` operation.
struct HasValueOpConversion : public FIROpConversion<fir::HasValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::HasValueOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Lower `fir.global` operation to `llvm.global` operation.
/// `fir.insert_on_range` operations are replaced with constant dense attribute
/// if they are applied on the full range.
struct GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalOp global, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = convertType(global.getType());
    if (global.getType().isa<fir::BoxType>())
      tyAttr = tyAttr.cast<mlir::LLVM::LLVMPointerType>().getElementType();
    auto loc = global.getLoc();
    mlir::Attribute initAttr{};
    if (global.initVal())
      initAttr = global.initVal().getValue();
    auto linkage = convertLinkage(global.linkName());
    auto isConst = global.constant().hasValue();
    auto g = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, tyAttr, isConst, linkage, global.sym_name(), initAttr);
    auto &gr = g.getInitializerRegion();
    rewriter.inlineRegionBefore(global.region(), gr, gr.end());
    if (!gr.empty()) {
      // Replace insert_on_range with a constant dense attribute if the
      // initialization is on the full range.
      auto insertOnRangeOps = gr.front().getOps<fir::InsertOnRangeOp>();
      for (auto insertOp : insertOnRangeOps) {
        if (isFullRange(insertOp.coor(), insertOp.getType())) {
          auto seqTyAttr = convertType(insertOp.getType());
          auto *op = insertOp.val().getDefiningOp();
          auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op);
          if (!constant) {
            auto convertOp = mlir::dyn_cast<fir::ConvertOp>(op);
            if (!convertOp)
              continue;
            constant = cast<mlir::arith::ConstantOp>(
                convertOp.value().getDefiningOp());
          }
          mlir::Type vecType = mlir::VectorType::get(
              insertOp.getType().getShape(), constant.getType());
          auto denseAttr = mlir::DenseElementsAttr::get(
              vecType.cast<ShapedType>(), constant.value());
          rewriter.setInsertionPointAfter(insertOp);
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
              insertOp, seqTyAttr, denseAttr);
        }
      }
    }
    rewriter.eraseOp(global);
    return success();
  }

  bool isFullRange(mlir::ArrayAttr indexes, fir::SequenceType seqTy) const {
    auto extents = seqTy.getShape();
    if (indexes.size() / 2 != extents.size())
      return false;
    for (unsigned i = 0; i < indexes.size(); i += 2) {
      if (indexes[i].cast<IntegerAttr>().getInt() != 0)
        return false;
      if (indexes[i + 1].cast<IntegerAttr>().getInt() != extents[i / 2] - 1)
        return false;
    }
    return true;
  }

  // TODO: String comparaison should be avoided. Replace linkName with an
  // enumeration.
  mlir::LLVM::Linkage convertLinkage(Optional<StringRef> optLinkage) const {
    if (optLinkage.hasValue()) {
      auto name = optLinkage.getValue();
      if (name == "internal")
        return mlir::LLVM::Linkage::Internal;
      if (name == "linkonce")
        return mlir::LLVM::Linkage::Linkonce;
      if (name == "common")
        return mlir::LLVM::Linkage::Common;
      if (name == "weak")
        return mlir::LLVM::Linkage::Weak;
    }
    return mlir::LLVM::Linkage::External;
  }
};

template <typename OP>
void selectMatchAndRewrite(fir::LLVMTypeConverter &lowering, OP select,
                           typename OP::Adaptor adaptor,
                           mlir::ConversionPatternRewriter &rewriter) {
  unsigned conds = select.getNumConditions();
  auto cases = select.getCases().getValue();
  mlir::Value selector = adaptor.selector();
  auto loc = select.getLoc();
  assert(conds > 0 && "select must have cases");

  llvm::SmallVector<mlir::Block *> destinations;
  llvm::SmallVector<mlir::ValueRange> destinationsOperands;
  mlir::Block *defaultDestination;
  mlir::ValueRange defaultOperands;
  llvm::SmallVector<int32_t> caseValues;

  for (unsigned t = 0; t != conds; ++t) {
    mlir::Block *dest = select.getSuccessor(t);
    auto destOps = select.getSuccessorOperands(adaptor.getOperands(), t);
    const mlir::Attribute &attr = cases[t];
    if (auto intAttr = attr.template dyn_cast<mlir::IntegerAttr>()) {
      destinations.push_back(dest);
      destinationsOperands.push_back(destOps.hasValue() ? *destOps
                                                        : ValueRange());
      caseValues.push_back(intAttr.getInt());
      continue;
    }
    assert(attr.template dyn_cast_or_null<mlir::UnitAttr>());
    assert((t + 1 == conds) && "unit must be last");
    defaultDestination = dest;
    defaultOperands = destOps.hasValue() ? *destOps : ValueRange();
  }

  // LLVM::SwitchOp takes a i32 type for the selector.
  if (select.getSelector().getType() != rewriter.getI32Type())
    selector =
        rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), selector);

  rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
      select, selector,
      /*defaultDestination=*/defaultDestination,
      /*defaultOperands=*/defaultOperands,
      /*caseValues=*/caseValues,
      /*caseDestinations=*/destinations,
      /*caseOperands=*/destinationsOperands,
      /*branchWeights=*/ArrayRef<int32_t>());
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectOp>(lowerTy(), op, adaptor, rewriter);
    return success();
  }
};

/// `fir.load` --> `llvm.load`
struct LoadOpConversion : public FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::LoadOp load, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // fir.box is a special case because it is considered as an ssa values in
    // fir, but it is lowered as a pointer to a descriptor. So fir.ref<fir.box>
    // and fir.box end up being the same llvm types and loading a
    // fir.ref<fir.box> is actually a no op in LLVM.
    if (load.getType().isa<fir::BoxType>()) {
      rewriter.replaceOp(load, adaptor.getOperands()[0]);
    } else {
      mlir::Type ty = convertType(load.getType());
      ArrayRef<NamedAttribute> at = load->getAttrs();
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
          load, ty, adaptor.getOperands(), at);
    }
    return success();
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public FIROpConversion<fir::SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectRankOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectRankOp>(lowerTy(), op, adaptor, rewriter);
    return success();
  }
};

/// `fir.store` --> `llvm.store`
struct StoreOpConversion : public FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StoreOp store, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (store.value().getType().isa<fir::BoxType>()) {
      // fir.box value is actually in memory, load it first before storing it.
      mlir::Location loc = store.getLoc();
      mlir::Type boxPtrTy = adaptor.getOperands()[0].getType();
      auto val = rewriter.create<mlir::LLVM::LoadOp>(
          loc, boxPtrTy.cast<mlir::LLVM::LLVMPointerType>().getElementType(),
          adaptor.getOperands()[0]);
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
          store, val, adaptor.getOperands()[1]);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(
          store, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    }
    return success();
  }
};

/// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public FIROpConversion<fir::UndefOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UndefOp undef, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(
        undef, convertType(undef.getType()));
    return success();
  }
};

/// `fir.unreachable` --> `llvm.unreachable`
struct UnreachableOpConversion : public FIROpConversion<fir::UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnreachableOp unreach, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(unreach);
    return success();
  }
};

struct ZeroOpConversion : public FIROpConversion<fir::ZeroOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ZeroOp zero, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(zero.getType());
    if (ty.isa<mlir::LLVM::LLVMPointerType>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(zero, ty);
    } else if (ty.isa<mlir::IntegerType>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
          zero, ty, mlir::IntegerAttr::get(zero.getType(), 0));
    } else if (mlir::LLVM::isCompatibleFloatingPointType(ty)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
          zero, ty, mlir::FloatAttr::get(zero.getType(), 0.0));
    } else {
      // TODO: create ConstantAggregateZero for FIR aggregate/array types.
      return rewriter.notifyMatchFailure(
          zero,
          "conversion of fir.zero with aggregate type not implemented yet");
    }
    return success();
  }
};

// Code shared between insert_value and extract_value Ops.
struct ValueOpCommon {
  // Translate the arguments pertaining to any multidimensional array to
  // row-major order for LLVM-IR.
  static void toRowMajor(SmallVectorImpl<mlir::Attribute> &attrs,
                         mlir::Type ty) {
    assert(ty && "type is null");
    const auto end = attrs.size();
    for (std::remove_const_t<decltype(end)> i = 0; i < end; ++i) {
      if (auto seq = ty.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
        const auto dim = getDimension(seq);
        if (dim > 1) {
          auto ub = std::min(i + dim, end);
          std::reverse(attrs.begin() + i, attrs.begin() + ub);
          i += dim - 1;
        }
        ty = getArrayElementType(seq);
      } else if (auto st = ty.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        ty = st.getBody()[attrs[i].cast<mlir::IntegerAttr>().getInt()];
      } else {
        llvm_unreachable("index into invalid type");
      }
    }
  }

  static llvm::SmallVector<mlir::Attribute>
  collectIndices(mlir::ConversionPatternRewriter &rewriter,
                 mlir::ArrayAttr arrAttr) {
    llvm::SmallVector<mlir::Attribute> attrs;
    for (auto i = arrAttr.begin(), e = arrAttr.end(); i != e; ++i) {
      if (i->isa<mlir::IntegerAttr>()) {
        attrs.push_back(*i);
      } else {
        auto fieldName = i->cast<mlir::StringAttr>().getValue();
        ++i;
        auto ty = i->cast<mlir::TypeAttr>().getValue();
        auto index = ty.cast<fir::RecordType>().getFieldIndex(fieldName);
        attrs.push_back(mlir::IntegerAttr::get(rewriter.getI32Type(), index));
      }
    }
    return attrs;
  }

private:
  static unsigned getDimension(mlir::LLVM::LLVMArrayType ty) {
    unsigned result = 1;
    for (auto eleTy = ty.getElementType().dyn_cast<mlir::LLVM::LLVMArrayType>();
         eleTy;
         eleTy = eleTy.getElementType().dyn_cast<mlir::LLVM::LLVMArrayType>())
      ++result;
    return result;
  }

  static mlir::Type getArrayElementType(mlir::LLVM::LLVMArrayType ty) {
    auto eleTy = ty.getElementType();
    while (auto arrTy = eleTy.dyn_cast<mlir::LLVM::LLVMArrayType>())
      eleTy = arrTy.getElementType();
    return eleTy;
  }
};

/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion
    : public FIROpAndTypeConversion<fir::ExtractValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::ExtractValueOp extractVal, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto attrs = collectIndices(rewriter, extractVal.coor());
    toRowMajor(attrs, adaptor.getOperands()[0].getType());
    auto position = mlir::ArrayAttr::get(extractVal.getContext(), attrs);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        extractVal, ty, adaptor.getOperands()[0], position);
    return success();
  }
};

/// InsertValue is the generalized instruction for the composition of new
/// aggregate type values.
struct InsertValueOpConversion
    : public FIROpAndTypeConversion<fir::InsertValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::InsertValueOp insertVal, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto attrs = collectIndices(rewriter, insertVal.coor());
    toRowMajor(attrs, adaptor.getOperands()[0].getType());
    auto position = mlir::ArrayAttr::get(insertVal.getContext(), attrs);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        insertVal, ty, adaptor.getOperands()[0], adaptor.getOperands()[1],
        position);
    return success();
  }
};

/// InsertOnRange inserts a value into a sequence over a range of offsets.
struct InsertOnRangeOpConversion
    : public FIROpAndTypeConversion<fir::InsertOnRangeOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  // Increments an array of subscripts in a row major fasion.
  void incrementSubscripts(const SmallVector<uint64_t> &dims,
                           SmallVector<uint64_t> &subscripts) const {
    for (size_t i = dims.size(); i > 0; --i) {
      if (++subscripts[i - 1] < dims[i - 1]) {
        return;
      }
      subscripts[i - 1] = 0;
    }
  }

  mlir::LogicalResult
  doRewrite(fir::InsertOnRangeOp range, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<uint64_t> dims;
    auto type = adaptor.getOperands()[0].getType();

    // Iteratively extract the array dimensions from the type.
    while (auto t = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
      dims.push_back(t.getNumElements());
      type = t.getElementType();
    }

    SmallVector<uint64_t> lBounds;
    SmallVector<uint64_t> uBounds;

    // Extract integer value from the attribute
    SmallVector<int64_t> coordinates = llvm::to_vector<4>(
        llvm::map_range(range.coor(), [](Attribute a) -> int64_t {
          return a.cast<IntegerAttr>().getInt();
        }));

    // Unzip the upper and lower bound and convert to a row major format.
    for (auto i = coordinates.rbegin(), e = coordinates.rend(); i != e; ++i) {
      uBounds.push_back(*i++);
      lBounds.push_back(*i);
    }

    auto &subscripts = lBounds;
    auto loc = range.getLoc();
    mlir::Value lastOp = adaptor.getOperands()[0];
    mlir::Value insertVal = adaptor.getOperands()[1];

    auto i64Ty = rewriter.getI64Type();
    while (subscripts != uBounds) {
      // Convert uint64_t's to Attribute's.
      SmallVector<mlir::Attribute> subscriptAttrs;
      for (const auto &subscript : subscripts)
        subscriptAttrs.push_back(IntegerAttr::get(i64Ty, subscript));
      lastOp = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, ty, lastOp, insertVal,
          ArrayAttr::get(range.getContext(), subscriptAttrs));

      incrementSubscripts(dims, subscripts);
    }

    // Convert uint64_t's to Attribute's.
    SmallVector<mlir::Attribute> subscriptAttrs;
    for (const auto &subscript : subscripts)
      subscriptAttrs.push_back(
          IntegerAttr::get(rewriter.getI64Type(), subscript));
    mlir::ArrayRef<mlir::Attribute> arrayRef(subscriptAttrs);

    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        range, ty, lastOp, insertVal,
        ArrayAttr::get(range.getContext(), arrayRef));

    return success();
  }
};

//
// Primitive operations on Complex types
//

/// Generate inline code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
mlir::LLVM::InsertValueOp complexSum(OPTY sumop, mlir::ValueRange opnds,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     fir::LLVMTypeConverter &lowering) {
  mlir::Value a = opnds[0];
  mlir::Value b = opnds[1];
  auto loc = sumop.getLoc();
  auto ctx = sumop.getContext();
  auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
  auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
  mlir::Type eleTy = lowering.convertType(getComplexEleTy(sumop.getType()));
  mlir::Type ty = lowering.convertType(sumop.getType());
  auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c0);
  auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c1);
  auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c0);
  auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c1);
  auto rx = rewriter.create<LLVMOP>(loc, eleTy, x0, x1);
  auto ry = rewriter.create<LLVMOP>(loc, eleTy, y0, y1);
  auto r0 = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
  auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r0, rx, c0);
  return rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r1, ry, c1);
}

struct AddcOpConversion : public FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddcOp addc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) + (x' + iy')
    // result: (x + x') + i(y + y')
    auto r = complexSum<mlir::LLVM::FAddOp>(addc, adaptor.getOperands(),
                                            rewriter, lowerTy());
    rewriter.replaceOp(addc, r.getResult());
    return success();
  }
};

struct SubcOpConversion : public FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SubcOp subc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) - (x' + iy')
    // result: (x - x') + i(y - y')
    auto r = complexSum<mlir::LLVM::FSubOp>(subc, adaptor.getOperands(),
                                            rewriter, lowerTy());
    rewriter.replaceOp(subc, r.getResult());
    return success();
  }
};

/// Inlined complex multiply
struct MulcOpConversion : public FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::MulcOp mulc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: Can we use a call to __muldc3 ?
    // given: (x + iy) * (x' + iy')
    // result: (xx'-yy')+i(xy'+yx')
    mlir::Value a = adaptor.getOperands()[0];
    mlir::Value b = adaptor.getOperands()[1];
    auto loc = mulc.getLoc();
    auto *ctx = mulc.getContext();
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    mlir::Type eleTy = convertType(getComplexEleTy(mulc.getType()));
    mlir::Type ty = convertType(mulc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1);
    auto ri = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xy, yx);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1);
    auto rr = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, xx, yy);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r1, ri, c1);
    rewriter.replaceOp(mulc, r0.getResult());
    return success();
  }
};

/// Inlined complex division
struct DivcOpConversion : public FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DivcOp divc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: Can we use a call to __divdc3 instead?
    // Just generate inline code for now.
    // given: (x + iy) / (x' + iy')
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    mlir::Value a = adaptor.getOperands()[0];
    mlir::Value b = adaptor.getOperands()[1];
    auto loc = divc.getLoc();
    auto *ctx = divc.getContext();
    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    mlir::Type eleTy = convertType(getComplexEleTy(divc.getType()));
    mlir::Type ty = convertType(divc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, a, c1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, b, c1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1);
    auto x1x1 = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x1, x1);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1);
    auto y1y1 = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y1, y1);
    auto d = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, x1x1, y1y1);
    auto rrn = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xx, yy);
    auto rin = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, yx, xy);
    auto rr = rewriter.create<mlir::LLVM::FDivOp>(loc, eleTy, rrn, d);
    auto ri = rewriter.create<mlir::LLVM::FDivOp>(loc, eleTy, rin, d);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r1, ri, c1);
    rewriter.replaceOp(divc, r0.getResult());
    return success();
  }
};

/// Inlined complex negation
struct NegcOpConversion : public FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NegcOp neg, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // given: -(x + iy)
    // result: -x - iy
    auto *ctxt = neg.getContext();
    auto eleTy = convertType(getComplexEleTy(neg.getType()));
    auto ty = convertType(neg.getType());
    auto loc = neg.getLoc();
    mlir::Value o0 = adaptor.getOperands()[0];
    auto c0 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(1));
    auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, o0, c0);
    auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, eleTy, o0, c1);
    auto nrp = rewriter.create<mlir::LLVM::FNegOp>(loc, eleTy, rp);
    auto nip = rewriter.create<mlir::LLVM::FNegOp>(loc, eleTy, ip);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, o0, nrp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(neg, ty, r, nip, c1);
    return success();
  }
};

} // namespace

namespace {
/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect. An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
///
/// This pass is not complete yet. We are upstreaming it in small patches.
class FIRToLLVMLowering : public fir::FIRToLLVMLoweringBase<FIRToLLVMLowering> {
public:
  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto mod = getModule();
    if (!forcedTargetTriple.empty()) {
      fir::setTargetTriple(mod, forcedTargetTriple);
    }

    auto *context = getModule().getContext();
    fir::LLVMTypeConverter typeConverter{getModule()};
    mlir::OwningRewritePatternList pattern(context);
    pattern.insert<
        AddcOpConversion, AddrOfOpConversion, BoxAddrOpConversion,
        BoxDimsOpConversion, BoxEleSizeOpConversion, BoxRankOpConversion,
        CallOpConversion, ConvertOpConversion, DivcOpConversion,
        ExtractValueOpConversion, HasValueOpConversion, GlobalOpConversion,
        InsertOnRangeOpConversion, InsertValueOpConversion, LoadOpConversion,
        NegcOpConversion, MulcOpConversion, SelectOpConversion,
        SelectRankOpConversion, StoreOpConversion, SubcOpConversion,
        UndefOpConversion, UnreachableOpConversion, ZeroOpConversion>(
        typeConverter);
    mlir::populateStdToLLVMConversionPatterns(typeConverter, pattern);
    mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                            pattern);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // required NOPs for applying a full conversion
    target.addLegalOp<mlir::ModuleOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyFullConversion(getModule(), target,
                                               std::move(pattern)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> fir::createFIRToLLVMPass() {
  return std::make_unique<FIRToLLVMLowering>();
}
