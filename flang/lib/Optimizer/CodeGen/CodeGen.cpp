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
#include "flang/ISO_Fortran_binding.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

#define DEBUG_TYPE "flang-codegen"

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "TypeConverter.h"

// TODO: This should really be recovered from the specified target.
static constexpr unsigned defaultAlign = 8;

/// `fir.box` attribute values as defined for CFI_attribute_t in
/// flang/ISO_Fortran_binding.h.
static constexpr unsigned kAttrPointer = CFI_attribute_pointer;
static constexpr unsigned kAttrAllocatable = CFI_attribute_allocatable;

static mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::Type ity,
                 mlir::ConversionPatternRewriter &rewriter,
                 std::int64_t offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

static Block *createBlock(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Block *insertBefore) {
  assert(insertBefore && "expected valid insertion block");
  return rewriter.createBlock(insertBefore->getParent(),
                              mlir::Region::iterator(insertBefore));
}

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
  genI32Constant(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                 int value) const {
    mlir::Type i32Ty = rewriter.getI32Type();
    mlir::IntegerAttr attr = rewriter.getI32IntegerAttr(value);
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Ty, attr);
  }

  mlir::LLVM::ConstantOp
  genConstantOffset(mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    int offset) const {
    mlir::Type ity = lowerTy().offsetType();
    mlir::IntegerAttr cattr = rewriter.getI32IntegerAttr(offset);
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
  }

  /// Construct code sequence to extract the specifc value from a `fir.box`.
  mlir::Value getValueFromBox(mlir::Location loc, mlir::Value box,
                              mlir::Type resultTy,
                              mlir::ConversionPatternRewriter &rewriter,
                              unsigned boxValue) const {
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    mlir::LLVM::ConstantOp cValuePos =
        genConstantOffset(loc, rewriter, boxValue);
    auto pty = mlir::LLVM::LLVMPointerType::get(resultTy);
    auto p = rewriter.create<mlir::LLVM::GEPOp>(
        loc, pty, mlir::ValueRange{box, c0, cValuePos});
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

  // Load the attribute from the \p box and perform a check against \p maskValue
  // The final comparison is implemented as `(attribute & maskValue) != 0`.
  mlir::Value genBoxAttributeCheck(mlir::Location loc, mlir::Value box,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   unsigned maskValue) const {
    mlir::Type attrTy = rewriter.getI32Type();
    mlir::Value attribute =
        getValueFromBox(loc, box, attrTy, rewriter, kAttributePosInBox);
    mlir::LLVM::ConstantOp attrMask =
        genConstantOffset(loc, rewriter, maskValue);
    auto maskRes =
        rewriter.create<mlir::LLVM::AndOp>(loc, attrTy, attribute, attrMask);
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    return rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, maskRes, c0);
  }

  // Get the element type given an LLVM type that is of the form
  // [llvm.ptr](array|struct|vector)+ and the provided indexes.
  static mlir::Type getBoxEleTy(mlir::Type type,
                                llvm::ArrayRef<unsigned> indexes) {
    if (auto t = type.dyn_cast<mlir::LLVM::LLVMPointerType>())
      type = t.getElementType();
    for (auto i : indexes) {
      if (auto t = type.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        assert(!t.isOpaque() && i < t.getBody().size());
        type = t.getBody()[i];
      } else if (auto t = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
        type = t.getElementType();
      } else if (auto t = type.dyn_cast<mlir::VectorType>()) {
        type = t.getElementType();
      } else {
        fir::emitFatalError(mlir::UnknownLoc::get(type.getContext()),
                            "request for invalid box element type");
      }
    }
    return type;
  }

  template <typename... ARGS>
  mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::Type ty,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value base, ARGS... args) const {
    SmallVector<mlir::Value> cv{args...};
    return rewriter.create<mlir::LLVM::GEPOp>(loc, ty, base, cv);
  }

  /// Perform an extension or truncation as needed on an integer value. Lowering
  /// to the specific target may involve some sign-extending or truncation of
  /// values, particularly to fit them from abstract box types to the
  /// appropriate reified structures.
  mlir::Value integerCast(mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::Type ty, mlir::Value val) const {
    auto valTy = val.getType();
    // If the value was not yet lowered, lower its type so that it can
    // be used in getPrimitiveTypeSizeInBits.
    if (!valTy.isa<mlir::IntegerType>())
      valTy = convertType(valTy);
    auto toSize = mlir::LLVM::getPrimitiveTypeSizeInBits(ty);
    auto fromSize = mlir::LLVM::getPrimitiveTypeSizeInBits(valTy);
    if (toSize < fromSize)
      return rewriter.create<mlir::LLVM::TruncOp>(loc, ty, val);
    if (toSize > fromSize)
      return rewriter.create<mlir::LLVM::SExtOp>(loc, ty, val);
    return val;
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

/// Create value signaling an absent optional argument in a call, e.g.
/// `fir.absent !fir.ref<i64>` -->  `llvm.mlir.null : !llvm.ptr<i64>`
struct AbsentOpConversion : public FIROpConversion<fir::AbsentOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AbsentOp absent, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = convertType(absent.getType());
    mlir::Location loc = absent.getLoc();

    if (absent.getType().isa<fir::BoxCharType>()) {
      auto structTy = ty.cast<mlir::LLVM::LLVMStructType>();
      assert(!structTy.isOpaque() && !structTy.getBody().empty());
      auto undefStruct = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
      auto nullField =
          rewriter.create<mlir::LLVM::NullOp>(loc, structTy.getBody()[0]);
      mlir::MLIRContext *ctx = absent.getContext();
      auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
          absent, ty, undefStruct, nullField, c0);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(absent, ty);
    }
    return success();
  }
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
} // namespace

/// Lookup the function to compute the memory size of this parametric derived
/// type. The size of the object may depend on the LEN type parameters of the
/// derived type.
static mlir::LLVM::LLVMFuncOp
getDependentTypeMemSizeFn(fir::RecordType recTy, fir::AllocaOp op,
                          mlir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  std::string name = recTy.getName().str() + "P.mem.size";
  return module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name);
}

namespace {
/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocaOp alloc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    auto loc = alloc.getLoc();
    mlir::Type ity = lowerTy().indexType();
    unsigned i = 0;
    mlir::Value size = genConstantIndex(loc, ity, rewriter, 1).getResult();
    mlir::Type ty = convertType(alloc.getType());
    mlir::Type resultTy = ty;
    if (alloc.hasLenParams()) {
      unsigned end = alloc.numLenParams();
      llvm::SmallVector<mlir::Value> lenParams;
      for (; i < end; ++i)
        lenParams.push_back(operands[i]);
      mlir::Type scalarType = fir::unwrapSequenceType(alloc.getInType());
      if (auto chrTy = scalarType.dyn_cast<fir::CharacterType>()) {
        fir::CharacterType rawCharTy = fir::CharacterType::getUnknownLen(
            chrTy.getContext(), chrTy.getFKind());
        ty = mlir::LLVM::LLVMPointerType::get(convertType(rawCharTy));
        assert(end == 1);
        size = integerCast(loc, rewriter, ity, lenParams[0]);
      } else if (auto recTy = scalarType.dyn_cast<fir::RecordType>()) {
        mlir::LLVM::LLVMFuncOp memSizeFn =
            getDependentTypeMemSizeFn(recTy, alloc, rewriter);
        if (!memSizeFn)
          emitError(loc, "did not find allocation function");
        mlir::NamedAttribute attr = rewriter.getNamedAttr(
            "callee", mlir::SymbolRefAttr::get(memSizeFn));
        auto call = rewriter.create<mlir::LLVM::CallOp>(
            loc, ity, lenParams, llvm::ArrayRef<mlir::NamedAttribute>{attr});
        size = call.getResult(0);
        ty = mlir::LLVM::LLVMPointerType::get(
            mlir::IntegerType::get(alloc.getContext(), 8));
      } else {
        return emitError(loc, "unexpected type ")
               << scalarType << " with type parameters";
      }
    }
    if (alloc.hasShapeOperands()) {
      mlir::Type allocEleTy = fir::unwrapRefType(alloc.getType());
      // Scale the size by constant factors encoded in the array type.
      if (auto seqTy = allocEleTy.dyn_cast<fir::SequenceType>()) {
        fir::SequenceType::Extent constSize = 1;
        for (auto extent : seqTy.getShape())
          if (extent != fir::SequenceType::getUnknownExtent())
            constSize *= extent;
        mlir::Value constVal{
            genConstantIndex(loc, ity, rewriter, constSize).getResult()};
        size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, constVal);
      }
      unsigned end = operands.size();
      for (; i < end; ++i)
        size = rewriter.create<mlir::LLVM::MulOp>(
            loc, ity, size, integerCast(loc, rewriter, ity, operands[i]));
    }
    if (ty == resultTy) {
      // Do not emit the bitcast if ty and resultTy are the same.
      rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(alloc, ty, size,
                                                        alloc->getAttrs());
    } else {
      auto al = rewriter.create<mlir::LLVM::AllocaOp>(loc, ty, size,
                                                      alloc->getAttrs());
      rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(alloc, resultTy, al);
    }
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
    auto elemSize = getValueFromBox(loc, a, ty, rewriter, kElemLenPosInBox);
    rewriter.replaceOp(boxelesz, elemSize);
    return success();
  }
};

/// Lower `fir.box_isalloc` to a sequence of operations to determine if the
/// boxed value was from an ALLOCATABLE entity.
struct BoxIsAllocOpConversion : public FIROpConversion<fir::BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsAllocOp boxisalloc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxisalloc.getLoc();
    mlir::Value check =
        genBoxAttributeCheck(loc, box, rewriter, kAttrAllocatable);
    rewriter.replaceOp(boxisalloc, check);
    return success();
  }
};

/// Lower `fir.box_isarray` to a sequence of operations to determine if the
/// boxed is an array.
struct BoxIsArrayOpConversion : public FIROpConversion<fir::BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsArrayOp boxisarray, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxisarray.getLoc();
    auto rank =
        getValueFromBox(loc, a, rewriter.getI32Type(), rewriter, kRankPosInBox);
    auto c0 = genConstantOffset(loc, rewriter, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisarray, mlir::LLVM::ICmpPredicate::ne, rank, c0);
    return success();
  }
};

/// Lower `fir.box_isptr` to a sequence of operations to determined if the
/// boxed value was from a POINTER entity.
struct BoxIsPtrOpConversion : public FIROpConversion<fir::BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxIsPtrOp boxisptr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxisptr.getLoc();
    mlir::Value check = genBoxAttributeCheck(loc, box, rewriter, kAttrPointer);
    rewriter.replaceOp(boxisptr, check);
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
    auto result = getValueFromBox(loc, a, ty, rewriter, kRankPosInBox);
    rewriter.replaceOp(boxrank, result);
    return success();
  }
};

/// Lower `fir.string_lit` to LLVM IR dialect operation.
struct StringLitOpConversion : public FIROpConversion<fir::StringLitOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StringLitOp constop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(constop.getType());
    auto attr = constop.getValue();
    if (attr.isa<mlir::StringAttr>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, attr);
      return success();
    }

    auto arr = attr.cast<mlir::ArrayAttr>();
    auto charTy = constop.getType().cast<fir::CharacterType>();
    unsigned bits = lowerTy().characterBitsize(charTy);
    mlir::Type intTy = rewriter.getIntegerType(bits);
    auto attrs = llvm::map_range(
        arr.getValue(), [intTy, bits](mlir::Attribute attr) -> Attribute {
          return mlir::IntegerAttr::get(
              intTy,
              attr.cast<mlir::IntegerAttr>().getValue().sextOrTrunc(bits));
        });
    mlir::Type vecType = mlir::VectorType::get(arr.size(), intTy);
    auto denseAttr = mlir::DenseElementsAttr::get(
        vecType.cast<mlir::ShapedType>(), llvm::to_vector<8>(attrs));
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(constop, ty,
                                                         denseAttr);
    return success();
  }
};

/// Lower `fir.boxproc_host` operation. Extracts the host pointer from the
/// boxproc.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct BoxProcHostOpConversion : public FIROpConversion<fir::BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxProcHostOp boxprochost, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        boxprochost, "fir.boxproc_host codegen is not implemented yet");
  }
};

/// Lower `fir.box_tdesc` to the sequence of operations to extract the type
/// descriptor from the box.
struct BoxTypeDescOpConversion : public FIROpConversion<fir::BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxTypeDescOp boxtypedesc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxtypedesc.getLoc();
    mlir::Type typeTy =
        fir::getDescFieldTypeModel<kTypePosInBox>()(boxtypedesc.getContext());
    auto result = getValueFromBox(loc, box, typeTy, rewriter, kTypePosInBox);
    auto typePtrTy = mlir::LLVM::LLVMPointerType::get(typeTy);
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(boxtypedesc, typePtrTy,
                                                        result);
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

/// Compare complex values
///
/// Per 10.1, the only comparisons available are .EQ. (oeq) and .NE. (une).
///
/// For completeness, all other comparison are done on the real component only.
struct CmpcOpConversion : public FIROpConversion<fir::CmpcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CmpcOp cmp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::MLIRContext *ctxt = cmp.getContext();
    mlir::Type eleTy = convertType(getComplexEleTy(cmp.lhs().getType()));
    mlir::Type resTy = convertType(cmp.getType());
    mlir::Location loc = cmp.getLoc();
    auto pos0 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(0));
    SmallVector<mlir::Value, 2> rp{rewriter.create<mlir::LLVM::ExtractValueOp>(
                                       loc, eleTy, operands[0], pos0),
                                   rewriter.create<mlir::LLVM::ExtractValueOp>(
                                       loc, eleTy, operands[1], pos0)};
    auto rcp =
        rewriter.create<mlir::LLVM::FCmpOp>(loc, resTy, rp, cmp->getAttrs());
    auto pos1 = mlir::ArrayAttr::get(ctxt, rewriter.getI32IntegerAttr(1));
    SmallVector<mlir::Value, 2> ip{rewriter.create<mlir::LLVM::ExtractValueOp>(
                                       loc, eleTy, operands[0], pos1),
                                   rewriter.create<mlir::LLVM::ExtractValueOp>(
                                       loc, eleTy, operands[1], pos1)};
    auto icp =
        rewriter.create<mlir::LLVM::FCmpOp>(loc, resTy, ip, cmp->getAttrs());
    SmallVector<mlir::Value, 2> cp{rcp, icp};
    switch (cmp.getPredicate()) {
    case mlir::arith::CmpFPredicate::OEQ: // .EQ.
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(cmp, resTy, cp);
      break;
    case mlir::arith::CmpFPredicate::UNE: // .NE.
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(cmp, resTy, cp);
      break;
    default:
      rewriter.replaceOp(cmp, rcp.getResult());
      break;
    }
    return success();
  }
};

/// Lower complex constants
struct ConstcOpConversion : public FIROpConversion<fir::ConstcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ConstcOp conc, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = conc.getLoc();
    mlir::MLIRContext *ctx = conc.getContext();
    mlir::Type ty = convertType(conc.getType());
    mlir::Type ety = convertType(getComplexEleTy(conc.getType()));
    auto realFloatAttr = mlir::FloatAttr::get(ety, getValue(conc.getReal()));
    auto realPart =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, realFloatAttr);
    auto imFloatAttr = mlir::FloatAttr::get(ety, getValue(conc.getImaginary()));
    auto imPart =
        rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, imFloatAttr);
    auto realIndex = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto imIndex = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto undef = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto setReal = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, ty, undef, realPart, realIndex);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(conc, ty, setReal,
                                                           imPart, imIndex);
    return success();
  }

  inline APFloat getValue(mlir::Attribute attr) const {
    return attr.cast<fir::RealAttr>().getValue();
  }
};

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

/// Lower `fir.dispatch` operation. A virtual call to a method in a dispatch
/// table.
struct DispatchOpConversion : public FIROpConversion<fir::DispatchOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchOp dispatch, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        dispatch, "fir.dispatch codegen is not implemented yet");
  }
};

/// Lower `fir.dispatch_table` operation. The dispatch table for a Fortran
/// derived type.
struct DispatchTableOpConversion
    : public FIROpConversion<fir::DispatchTableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchTableOp dispTab, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        dispTab, "fir.dispatch_table codegen is not implemented yet");
  }
};

/// Lower `fir.dt_entry` operation. An entry in a dispatch table; binds a
/// method-name to a function.
struct DTEntryOpConversion : public FIROpConversion<fir::DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DTEntryOp dtEnt, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        dtEnt, "fir.dt_entry codegen is not implemented yet");
  }
};

/// Lower `fir.global_len` operation.
struct GlobalLenOpConversion : public FIROpConversion<fir::GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalLenOp globalLen, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        globalLen, "fir.global_len codegen is not implemented yet");
  }
};

/// Lower `fir.gentypedesc` to a global constant.
struct GenTypeDescOpConversion : public FIROpConversion<fir::GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GenTypeDescOp gentypedesc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        gentypedesc, "fir.fir.gentypedesc codegen is not implemented yet");
  }
};

/// Convert `fir.end`
struct FirEndOpConversion : public FIROpConversion<fir::FirEndOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FirEndOp firEnd, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        firEnd, "fir.end codegen is not implemented yet");
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

void genCondBrOp(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                 Optional<mlir::ValueRange> destOps,
                 mlir::ConversionPatternRewriter &rewriter,
                 mlir::Block *newBlock) {
  if (destOps.hasValue())
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, destOps.getValue(),
                                          newBlock, mlir::ValueRange());
  else
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, newBlock);
}

template <typename A, typename B>
void genBrOp(A caseOp, mlir::Block *dest, Optional<B> destOps,
             mlir::ConversionPatternRewriter &rewriter) {
  if (destOps.hasValue())
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, destOps.getValue(),
                                                  dest);
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, llvm::None, dest);
}

void genCaseLadderStep(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                       Optional<mlir::ValueRange> destOps,
                       mlir::ConversionPatternRewriter &rewriter) {
  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock = createBlock(rewriter, dest);
  rewriter.setInsertionPointToEnd(thisBlock);
  genCondBrOp(loc, cmp, dest, destOps, rewriter, newBlock);
  rewriter.setInsertionPointToEnd(newBlock);
}

/// Conversion of `fir.select_case`
///
/// The `fir.select_case` operation is converted to a if-then-else ladder.
/// Depending on the case condition type, one or several comparison and
/// conditional branching can be generated.
///
/// A a point value case such as `case(4)`, a lower bound case such as
/// `case(5:)` or an upper bound case such as `case(:3)` are converted to a
/// simple comparison between the selector value and the constant value in the
/// case. The block associated with the case condition is then executed if
/// the comparison succeed otherwise it branch to the next block with the
/// comparison for the the next case conditon.
///
/// A closed interval case condition such as `case(7:10)` is converted with a
/// first comparison and conditional branching for the lower bound. If
/// successful, it branch to a second block with the comparison for the
/// upper bound in the same case condition.
///
/// TODO: lowering of CHARACTER type cases is not handled yet.
struct SelectCaseOpConversion : public FIROpConversion<fir::SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectCaseOp caseOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    unsigned conds = caseOp.getNumConditions();
    llvm::ArrayRef<mlir::Attribute> cases = caseOp.getCases().getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    LLVM_ATTRIBUTE_UNUSED auto ty = caseOp.getSelector().getType();
    if (ty.isa<fir::CharacterType>())
      return rewriter.notifyMatchFailure(caseOp,
                                         "conversion of fir.select_case with "
                                         "character type not implemented yet");
    mlir::Value selector = caseOp.getSelector(adaptor.getOperands());
    auto loc = caseOp.getLoc();
    for (unsigned t = 0; t != conds; ++t) {
      mlir::Block *dest = caseOp.getSuccessor(t);
      llvm::Optional<mlir::ValueRange> destOps =
          caseOp.getSuccessorOperands(adaptor.getOperands(), t);
      llvm::Optional<mlir::ValueRange> cmpOps =
          *caseOp.getCompareOperands(adaptor.getOperands(), t);
      mlir::Value caseArg = *(cmpOps.getValue().begin());
      mlir::Attribute attr = cases[t];
      if (attr.isa<fir::PointIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, selector, caseArg);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::LowerBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg, selector);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::UpperBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (attr.isa<fir::ClosedIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg, selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = createBlock(rewriter, dest);
        auto *newBlock2 = createBlock(rewriter, dest);
        rewriter.setInsertionPointToEnd(thisBlock);
        rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, newBlock1, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock1);
        mlir::Value caseArg0 = *(cmpOps.getValue().begin() + 1);
        auto cmp0 = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg0);
        genCondBrOp(loc, cmp0, dest, destOps, rewriter, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(attr.isa<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      genBrOp(caseOp, dest, destOps, rewriter);
    }
    return success();
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

/// Lower `fir.no_reassoc` to LLVM IR dialect.
/// TODO: how do we want to enforce this in LLVM-IR? Can we manipulate the fast
/// math flags?
struct NoReassocOpConversion : public FIROpConversion<fir::NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::NoReassocOp noreassoc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(noreassoc, adaptor.getOperands()[0]);
    return success();
  }
};

/// Lower `fir.select_type` to LLVM IR dialect.
struct SelectTypeOpConversion : public FIROpConversion<fir::SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectTypeOp select, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        select, "fir.select_type codegen is not implemented yet");
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

/// Common base class for embox to descriptor conversion.
template <typename OP>
struct EmboxCommonConversion : public FIROpConversion<OP> {
  using FIROpConversion<OP>::FIROpConversion;

  // Find the LLVMFuncOp in whose entry block the alloca should be inserted.
  // The order to find the LLVMFuncOp is as follows:
  // 1. The parent operation of the current block if it is a LLVMFuncOp.
  // 2. The first ancestor that is a LLVMFuncOp.
  mlir::LLVM::LLVMFuncOp
  getFuncForAllocaInsert(mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
    return mlir::isa<mlir::LLVM::LLVMFuncOp>(parentOp)
               ? mlir::cast<mlir::LLVM::LLVMFuncOp>(parentOp)
               : parentOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  }

  // Generate an alloca of size 1 and type \p toTy.
  mlir::LLVM::AllocaOp
  genAllocaWithType(mlir::Location loc, mlir::Type toTy, unsigned alignment,
                    mlir::ConversionPatternRewriter &rewriter) const {
    auto thisPt = rewriter.saveInsertionPoint();
    mlir::LLVM::LLVMFuncOp func = getFuncForAllocaInsert(rewriter);
    rewriter.setInsertionPointToStart(&func.front());
    auto size = this->genI32Constant(loc, rewriter, 1);
    auto al = rewriter.create<mlir::LLVM::AllocaOp>(loc, toTy, size, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return al;
  }

  static int getCFIAttr(fir::BoxType boxTy) {
    auto eleTy = boxTy.getEleTy();
    if (eleTy.isa<fir::PointerType>())
      return CFI_attribute_pointer;
    if (eleTy.isa<fir::HeapType>())
      return CFI_attribute_allocatable;
    return CFI_attribute_other;
  }

  static fir::RecordType unwrapIfDerived(fir::BoxType boxTy) {
    return fir::unwrapSequenceType(fir::dyn_cast_ptrOrBoxEleTy(boxTy))
        .template dyn_cast<fir::RecordType>();
  }
  static bool isDerivedTypeWithLenParams(fir::BoxType boxTy) {
    auto recTy = unwrapIfDerived(boxTy);
    return recTy && recTy.getNumLenParams() > 0;
  }
  static bool isDerivedType(fir::BoxType boxTy) {
    return unwrapIfDerived(boxTy) != nullptr;
  }

  // Get the element size and CFI type code of the boxed value.
  std::tuple<mlir::Value, mlir::Value> getSizeAndTypeCode(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      mlir::Type boxEleTy, mlir::ValueRange lenParams = {}) const {
    auto doInteger =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode = fir::integerBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doLogical =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode = fir::logicalBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doFloat = [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      int typeCode = fir::realBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doComplex =
        [&](unsigned width) -> std::tuple<mlir::Value, mlir::Value> {
      auto typeCode = fir::complexBitsToTypeCode(width);
      return {this->genConstantOffset(loc, rewriter, width / 8 * 2),
              this->genConstantOffset(loc, rewriter, typeCode)};
    };
    auto doCharacter =
        [&](unsigned width,
            mlir::Value len) -> std::tuple<mlir::Value, mlir::Value> {
      auto typeCode = fir::characterBitsToTypeCode(width);
      auto typeCodeVal = this->genConstantOffset(loc, rewriter, typeCode);
      if (width == 8)
        return {len, typeCodeVal};
      auto byteWidth = this->genConstantOffset(loc, rewriter, width / 8);
      auto i64Ty = mlir::IntegerType::get(&this->lowerTy().getContext(), 64);
      auto size =
          rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, byteWidth, len);
      return {size, typeCodeVal};
    };
    auto getKindMap = [&]() -> fir::KindMapping & {
      return this->lowerTy().getKindMap();
    };
    // Pointer-like types.
    if (auto eleTy = fir::dyn_cast_ptrEleTy(boxEleTy))
      boxEleTy = eleTy;
    // Integer types.
    if (fir::isa_integer(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::IntegerType>())
        return doInteger(ty.getWidth());
      auto ty = boxEleTy.cast<fir::IntegerType>();
      return doInteger(getKindMap().getIntegerBitsize(ty.getFKind()));
    }
    // Floating point types.
    if (fir::isa_real(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::FloatType>())
        return doFloat(ty.getWidth());
      auto ty = boxEleTy.cast<fir::RealType>();
      return doFloat(getKindMap().getRealBitsize(ty.getFKind()));
    }
    // Complex types.
    if (fir::isa_complex(boxEleTy)) {
      if (auto ty = boxEleTy.dyn_cast<mlir::ComplexType>())
        return doComplex(
            ty.getElementType().cast<mlir::FloatType>().getWidth());
      auto ty = boxEleTy.cast<fir::ComplexType>();
      return doComplex(getKindMap().getRealBitsize(ty.getFKind()));
    }
    // Character types.
    if (auto ty = boxEleTy.dyn_cast<fir::CharacterType>()) {
      auto charWidth = getKindMap().getCharacterBitsize(ty.getFKind());
      if (ty.getLen() != fir::CharacterType::unknownLen()) {
        auto len = this->genConstantOffset(loc, rewriter, ty.getLen());
        return doCharacter(charWidth, len);
      }
      assert(!lenParams.empty());
      return doCharacter(charWidth, lenParams.back());
    }
    // Logical type.
    if (auto ty = boxEleTy.dyn_cast<fir::LogicalType>())
      return doLogical(getKindMap().getLogicalBitsize(ty.getFKind()));
    // Array types.
    if (auto seqTy = boxEleTy.dyn_cast<fir::SequenceType>())
      return getSizeAndTypeCode(loc, rewriter, seqTy.getEleTy(), lenParams);
    // Derived-type types.
    if (boxEleTy.isa<fir::RecordType>()) {
      auto ptrTy = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(boxEleTy));
      auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, ptrTy);
      auto one =
          genConstantIndex(loc, this->lowerTy().offsetType(), rewriter, 1);
      auto gep = rewriter.create<mlir::LLVM::GEPOp>(
          loc, ptrTy, mlir::ValueRange{nullPtr, one});
      auto eleSize = rewriter.create<mlir::LLVM::PtrToIntOp>(
          loc, this->lowerTy().indexType(), gep);
      return {eleSize,
              this->genConstantOffset(loc, rewriter, fir::derivedToTypeCode())};
    }
    // Reference type.
    if (fir::isa_ref_type(boxEleTy)) {
      // FIXME: use the target pointer size rather than sizeof(void*)
      return {this->genConstantOffset(loc, rewriter, sizeof(void *)),
              this->genConstantOffset(loc, rewriter, CFI_type_cptr)};
    }
    fir::emitFatalError(loc, "unhandled type in fir.box code generation");
  }

  /// Basic pattern to write a field in the descriptor
  mlir::Value insertField(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, mlir::Value dest,
                          ArrayRef<unsigned> fldIndexes, mlir::Value value,
                          bool bitcast = false) const {
    auto boxTy = dest.getType();
    auto fldTy = this->getBoxEleTy(boxTy, fldIndexes);
    if (bitcast)
      value = rewriter.create<mlir::LLVM::BitcastOp>(loc, fldTy, value);
    else
      value = this->integerCast(loc, rewriter, fldTy, value);
    SmallVector<mlir::Attribute, 2> attrs;
    for (auto i : fldIndexes)
      attrs.push_back(rewriter.getI32IntegerAttr(i));
    auto indexesAttr = mlir::ArrayAttr::get(rewriter.getContext(), attrs);
    return rewriter.create<mlir::LLVM::InsertValueOp>(loc, boxTy, dest, value,
                                                      indexesAttr);
  }

  inline mlir::Value
  insertBaseAddress(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, mlir::Value dest,
                    mlir::Value base) const {
    return insertField(rewriter, loc, dest, {0}, base, /*bitCast=*/true);
  }

  /// Get the address of the type descriptor global variable that was created by
  /// lowering for derived type \p recType.
  template <typename BOX>
  mlir::Value
  getTypeDescriptor(BOX box, mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, fir::RecordType recType) const {
    std::string name = recType.getLoweredName();
    auto module = box->template getParentOfType<mlir::ModuleOp>();
    if (auto global = module.template lookupSymbol<fir::GlobalOp>(name)) {
      auto ty = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(global.getType()));
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty,
                                                      global.sym_name());
    }
    if (auto global =
            module.template lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
      // The global may have already been translated to LLVM.
      auto ty = mlir::LLVM::LLVMPointerType::get(global.getType());
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty,
                                                      global.sym_name());
    }
    // The global does not exist in the current translation unit, but may be
    // defined elsewhere (e.g., type defined in a module).
    // For now, create a extern_weak symbol (will become nullptr if unresolved)
    // to support generating code without the front-end generated symbols.
    // These could be made available_externally to require the symbols to be
    // defined elsewhere and to cause link-time failure otherwise.
    auto i8Ty = rewriter.getIntegerType(8);
    mlir::OpBuilder modBuilder(module.getBodyRegion());
    // TODO: The symbol should be lowered to constant in lowering, they are read
    // only.
    modBuilder.create<mlir::LLVM::GlobalOp>(loc, i8Ty, /*isConstant=*/false,
                                            mlir::LLVM::Linkage::ExternWeak,
                                            name, mlir::Attribute{});
    auto ty = mlir::LLVM::LLVMPointerType::get(i8Ty);
    return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty, name);
  }

  template <typename BOX>
  std::tuple<fir::BoxType, mlir::Value, mlir::Value>
  consDescriptorPrefix(BOX box, mlir::ConversionPatternRewriter &rewriter,
                       unsigned rank, mlir::ValueRange lenParams) const {
    auto loc = box.getLoc();
    auto boxTy = box.getType().template dyn_cast<fir::BoxType>();
    auto convTy = this->lowerTy().convertBoxType(boxTy, rank);
    auto llvmBoxPtrTy = convTy.template cast<mlir::LLVM::LLVMPointerType>();
    auto llvmBoxTy = llvmBoxPtrTy.getElementType();
    mlir::Value descriptor =
        rewriter.create<mlir::LLVM::UndefOp>(loc, llvmBoxTy);

    llvm::SmallVector<mlir::Value> typeparams = lenParams;
    if constexpr (!std::is_same_v<BOX, fir::EmboxOp>) {
      if (!box.substr().empty() && fir::hasDynamicSize(boxTy.getEleTy()))
        typeparams.push_back(box.substr()[1]);
    }

    // Write each of the fields with the appropriate values
    auto [eleSize, cfiTy] =
        getSizeAndTypeCode(loc, rewriter, boxTy.getEleTy(), typeparams);
    descriptor =
        insertField(rewriter, loc, descriptor, {kElemLenPosInBox}, eleSize);
    descriptor = insertField(rewriter, loc, descriptor, {kVersionPosInBox},
                             this->genI32Constant(loc, rewriter, CFI_VERSION));
    descriptor = insertField(rewriter, loc, descriptor, {kRankPosInBox},
                             this->genI32Constant(loc, rewriter, rank));
    descriptor = insertField(rewriter, loc, descriptor, {kTypePosInBox}, cfiTy);
    descriptor =
        insertField(rewriter, loc, descriptor, {kAttributePosInBox},
                    this->genI32Constant(loc, rewriter, getCFIAttr(boxTy)));
    const bool hasAddendum = isDerivedType(boxTy);
    descriptor =
        insertField(rewriter, loc, descriptor, {kF18AddendumPosInBox},
                    this->genI32Constant(loc, rewriter, hasAddendum ? 1 : 0));

    if (hasAddendum) {
      auto isArray =
          fir::dyn_cast_ptrOrBoxEleTy(boxTy).template isa<fir::SequenceType>();
      unsigned typeDescFieldId = isArray ? kOptTypePtrPosInBox : kDimsPosInBox;
      auto typeDesc =
          getTypeDescriptor(box, rewriter, loc, unwrapIfDerived(boxTy));
      descriptor =
          insertField(rewriter, loc, descriptor, {typeDescFieldId}, typeDesc,
                      /*bitCast=*/true);
    }

    return {boxTy, descriptor, eleSize};
  }

  /// If the embox is not in a globalOp body, allocate storage for the box;
  /// store the value inside and return the generated alloca. Return the input
  /// value otherwise.
  mlir::Value
  placeInMemoryIfNotGlobalInit(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value boxValue) const {
    auto *thisBlock = rewriter.getInsertionBlock();
    if (thisBlock && mlir::isa<mlir::LLVM::GlobalOp>(thisBlock->getParentOp()))
      return boxValue;
    auto boxPtrTy = mlir::LLVM::LLVMPointerType::get(boxValue.getType());
    auto alloca = genAllocaWithType(loc, boxPtrTy, defaultAlign, rewriter);
    rewriter.create<mlir::LLVM::StoreOp>(loc, boxValue, alloca);
    return alloca;
  }
};

/// Create a generic box on a memory reference. This conversions lowers the
/// abstract box to the appropriate, initialized descriptor.
struct EmboxOpConversion : public EmboxCommonConversion<fir::EmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxOp embox, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(!embox.getShape() && "There should be no dims on this embox op");
    auto [boxTy, dest, eleSize] =
        consDescriptorPrefix(embox, rewriter, /*rank=*/0,
                             /*lenParams=*/adaptor.getOperands().drop_front(1));
    dest = insertBaseAddress(rewriter, embox.getLoc(), dest,
                             adaptor.getOperands()[0]);
    if (isDerivedTypeWithLenParams(boxTy))
      return rewriter.notifyMatchFailure(
          embox, "fir.embox codegen of derived with length parameters not "
                 "implemented yet");
    auto result = placeInMemoryIfNotGlobalInit(rewriter, embox.getLoc(), dest);
    rewriter.replaceOp(embox, result);
    return success();
  }
};

/// Lower `fir.emboxproc` operation. Creates a procedure box.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct EmboxProcOpConversion : public FIROpConversion<fir::EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxProcOp emboxproc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        emboxproc, "fir.emboxproc codegen is not implemented yet");
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

/// Conversion pattern for operation that must be dead. The information in these
/// operations is used by other operation. At this point they should not have
/// anymore uses.
/// These operations are normally dead after the pre-codegen pass.
template <typename FromOp>
struct MustBeDeadConversion : public FIROpConversion<FromOp> {
  explicit MustBeDeadConversion(fir::LLVMTypeConverter &lowering)
      : FIROpConversion<FromOp>(lowering) {}
  using OpAdaptor = typename FromOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(FromOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (!op->getUses().empty())
      return rewriter.notifyMatchFailure(op, "op must be dead");
    rewriter.eraseOp(op);
    return success();
  }
};

struct ShapeOpConversion : public MustBeDeadConversion<fir::ShapeOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

struct ShapeShiftOpConversion : public MustBeDeadConversion<fir::ShapeShiftOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

struct ShiftOpConversion : public MustBeDeadConversion<fir::ShiftOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

struct SliceOpConversion : public MustBeDeadConversion<fir::SliceOp> {
  using MustBeDeadConversion::MustBeDeadConversion;
};

/// `fir.is_present` -->
/// ```
///  %0 = llvm.mlir.constant(0 : i64)
///  %1 = llvm.ptrtoint %0
///  %2 = llvm.icmp "ne" %1, %0 : i64
/// ```
struct IsPresentOpConversion : public FIROpConversion<fir::IsPresentOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::IsPresentOp isPresent, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type idxTy = lowerTy().indexType();
    mlir::Location loc = isPresent.getLoc();
    auto ptr = adaptor.getOperands()[0];

    if (isPresent.val().getType().isa<fir::BoxCharType>()) {
      auto structTy = ptr.getType().cast<mlir::LLVM::LLVMStructType>();
      assert(!structTy.isOpaque() && !structTy.getBody().empty());

      mlir::Type ty = structTy.getBody()[0];
      mlir::MLIRContext *ctx = isPresent.getContext();
      auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
      ptr = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, ptr, c0);
    }
    mlir::LLVM::ConstantOp c0 =
        genConstantIndex(isPresent.getLoc(), idxTy, rewriter, 0);
    auto addr = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, idxTy, ptr);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        isPresent, mlir::LLVM::ICmpPredicate::ne, addr, c0);

    return success();
  }
};

/// Convert `!fir.emboxchar<!fir.char<KIND, ?>, #n>` into a sequence of
/// instructions that generate `!llvm.struct<(ptr<ik>, i64)>`. The 1st element
/// in this struct is a pointer. Its type is determined from `KIND`. The 2nd
/// element is the length of the character buffer (`#n`).
struct EmboxCharOpConversion : public FIROpConversion<fir::EmboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxCharOp emboxChar, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    MLIRContext *ctx = emboxChar.getContext();

    mlir::Value charBuffer = operands[0];
    mlir::Value charBufferLen = operands[1];

    mlir::Location loc = emboxChar.getLoc();
    mlir::Type llvmStructTy = convertType(emboxChar.getType());
    auto llvmStruct = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmStructTy);

    mlir::Type lenTy =
        llvmStructTy.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
    mlir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, charBufferLen);

    auto c0 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(0));
    auto c1 = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(1));
    auto insertBufferOp = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, llvmStructTy, llvmStruct, charBuffer, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        emboxChar, llvmStructTy, insertBufferOp, lenAfterCast, c1);

    return success();
  }
};

/// Construct an `llvm.extractvalue` instruction. It will return value at
/// element \p x from  \p tuple.
mlir::LLVM::ExtractValueOp
genExtractValueWithIndex(mlir::Location loc, mlir::Value tuple, mlir::Type ty,
                         mlir::ConversionPatternRewriter &rewriter,
                         mlir::MLIRContext *ctx, int x) {
  auto cx = mlir::ArrayAttr::get(ctx, rewriter.getI32IntegerAttr(x));
  auto xty = ty.cast<mlir::LLVM::LLVMStructType>().getBody()[x];
  return rewriter.create<mlir::LLVM::ExtractValueOp>(loc, xty, tuple, cx);
}

/// Convert `!fir.boxchar_len` to  `!llvm.extractvalue` for the 2nd part of the
/// boxchar.
struct BoxCharLenOpConversion : public FIROpConversion<fir::BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxCharLenOp boxCharLen, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value boxChar = adaptor.getOperands()[0];
    mlir::Location loc = boxChar.getLoc();
    mlir::MLIRContext *ctx = boxChar.getContext();
    mlir::Type returnValTy = boxCharLen.getResult().getType();

    constexpr int boxcharLenIdx = 1;
    mlir::LLVM::ExtractValueOp len = genExtractValueWithIndex(
        loc, boxChar, boxChar.getType(), rewriter, ctx, boxcharLenIdx);
    mlir::Value lenAfterCast = integerCast(loc, rewriter, returnValTy, len);
    rewriter.replaceOp(boxCharLen, lenAfterCast);

    return success();
  }
};

/// Convert `fir.unboxchar` into two `llvm.extractvalue` instructions. One for
/// the character buffer and one for the buffer length.
struct UnboxCharOpConversion : public FIROpConversion<fir::UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxCharOp unboxchar, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = unboxchar.getContext();

    mlir::Type lenTy = convertType(unboxchar.getType(1));
    mlir::Value tuple = adaptor.getOperands()[0];
    mlir::Type tupleTy = tuple.getType();

    mlir::Location loc = unboxchar.getLoc();
    mlir::Value ptrToBuffer =
        genExtractValueWithIndex(loc, tuple, tupleTy, rewriter, ctx, 0);

    mlir::LLVM::ExtractValueOp len =
        genExtractValueWithIndex(loc, tuple, tupleTy, rewriter, ctx, 1);
    mlir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, len);

    rewriter.replaceOp(unboxchar,
                       ArrayRef<mlir::Value>{ptrToBuffer, lenAfterCast});
    return success();
  }
};

/// Lower `fir.unboxproc` operation. Unbox a procedure box value, yielding its
/// components.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct UnboxProcOpConversion : public FIROpConversion<fir::UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxProcOp unboxproc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return rewriter.notifyMatchFailure(
        unboxproc, "fir.unboxproc codegen is not implemented yet");
  }
};

/// Convert `fir.field_index`. The conversion depends on whether the size of
/// the record is static or dynamic.
struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  // NB: most field references should be resolved by this point
  mlir::LogicalResult
  matchAndRewrite(fir::FieldIndexOp field, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto recTy = field.on_type().cast<fir::RecordType>();
    unsigned index = recTy.getFieldIndex(field.field_id());

    if (!fir::hasDynamicSize(recTy)) {
      // Derived type has compile-time constant layout. Return index of the
      // component type in the parent type (to be used in GEP).
      rewriter.replaceOp(field, mlir::ValueRange{genConstantOffset(
                                    field.getLoc(), rewriter, index)});
      return success();
    }

    // Derived type has compile-time constant layout. Call the compiler
    // generated function to determine the byte offset of the field at runtime.
    // This returns a non-constant.
    FlatSymbolRefAttr symAttr = mlir::SymbolRefAttr::get(
        field.getContext(), getOffsetMethodName(recTy, field.field_id()));
    NamedAttribute callAttr = rewriter.getNamedAttr("callee", symAttr);
    NamedAttribute fieldAttr = rewriter.getNamedAttr(
        "field", mlir::IntegerAttr::get(lowerTy().indexType(), index));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        field, lowerTy().offsetType(), adaptor.getOperands(),
        llvm::ArrayRef<mlir::NamedAttribute>{callAttr, fieldAttr});
    return success();
  }

  // Re-Construct the name of the compiler generated method that calculates the
  // offset
  inline static std::string getOffsetMethodName(fir::RecordType recTy,
                                                llvm::StringRef field) {
    return recTy.getName().str() + "P." + field.str() + ".offset";
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
        AbsentOpConversion, AddcOpConversion, AddrOfOpConversion,
        AllocaOpConversion, BoxAddrOpConversion, BoxCharLenOpConversion,
        BoxDimsOpConversion, BoxEleSizeOpConversion, BoxIsAllocOpConversion,
        BoxIsArrayOpConversion, BoxIsPtrOpConversion, BoxProcHostOpConversion,
        BoxRankOpConversion, BoxTypeDescOpConversion, CallOpConversion,
        CmpcOpConversion, ConstcOpConversion, ConvertOpConversion,
        DispatchOpConversion, DispatchTableOpConversion, DTEntryOpConversion,
        DivcOpConversion, EmboxOpConversion, EmboxCharOpConversion,
        EmboxProcOpConversion, ExtractValueOpConversion, FieldIndexOpConversion,
        FirEndOpConversion, HasValueOpConversion, GenTypeDescOpConversion,
        GlobalLenOpConversion, GlobalOpConversion, InsertOnRangeOpConversion,
        InsertValueOpConversion, IsPresentOpConversion, LoadOpConversion,
        NegcOpConversion, NoReassocOpConversion, MulcOpConversion,
        SelectCaseOpConversion, SelectOpConversion, SelectRankOpConversion,
        SelectTypeOpConversion, ShapeOpConversion, ShapeShiftOpConversion,
        ShiftOpConversion, SliceOpConversion, StoreOpConversion,
        StringLitOpConversion, SubcOpConversion, UnboxCharOpConversion,
        UnboxProcOpConversion, UndefOpConversion, UnreachableOpConversion,
        ZeroOpConversion>(typeConverter);
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
