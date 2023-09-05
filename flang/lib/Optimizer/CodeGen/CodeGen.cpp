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

#include "CGOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Semantics/runtime-type-info.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
#define GEN_PASS_DEF_FIRTOLLVMLOWERING
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-codegen"

// fir::LLVMTypeConverter for converting to LLVM IR dialect types.
#include "flang/Optimizer/CodeGen/TypeConverter.h"

// TODO: This should really be recovered from the specified target.
static constexpr unsigned defaultAlign = 8;

/// `fir.box` attribute values as defined for CFI_attribute_t in
/// flang/ISO_Fortran_binding.h.
static constexpr unsigned kAttrPointer = CFI_attribute_pointer;
static constexpr unsigned kAttrAllocatable = CFI_attribute_allocatable;

static inline mlir::Type getVoidPtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
}

static mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::Type ity,
                 mlir::ConversionPatternRewriter &rewriter,
                 std::int64_t offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

static mlir::Block *createBlock(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Block *insertBefore) {
  assert(insertBefore && "expected valid insertion block");
  return rewriter.createBlock(insertBefore->getParent(),
                              mlir::Region::iterator(insertBefore));
}

/// Extract constant from a value if it is a result of one of the
/// ConstantOp operations, otherwise, return std::nullopt.
static std::optional<int64_t> getIfConstantIntValue(mlir::Value val) {
  if (!val || !val.dyn_cast<mlir::OpResult>())
    return {};

  mlir::Operation *defop = val.getDefiningOp();

  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defop))
    return constOp.value();
  if (auto llConstOp = mlir::dyn_cast<mlir::LLVM::ConstantOp>(defop))
    if (auto attr = llConstOp.getValue().dyn_cast<mlir::IntegerAttr>())
      return attr.getValue().getSExtValue();

  return {};
}

/// Extract constant from a value that must be the result of one of the
/// ConstantOp operations.
static int64_t getConstantIntValue(mlir::Value val) {
  if (auto constVal = getIfConstantIntValue(val))
    return *constVal;
  fir::emitFatalError(val.getLoc(), "must be a constant");
}

static unsigned getTypeDescFieldId(mlir::Type ty) {
  auto isArray = fir::dyn_cast_ptrOrBoxEleTy(ty).isa<fir::SequenceType>();
  return isArray ? kOptTypePtrPosInBox : kDimsPosInBox;
}

namespace {
/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::ConvertOpToLLVMPattern<FromOp> {
public:
  explicit FIROpConversion(const fir::LLVMTypeConverter &lowering,
                           const fir::FIRToLLVMPassOptions &options)
      : mlir::ConvertOpToLLVMPattern<FromOp>(lowering), options(options) {}

protected:
  mlir::Type convertType(mlir::Type ty) const {
    return lowerTy().convertType(ty);
  }
  mlir::Type voidPtrTy() const { return getVoidPtrType(); }

  mlir::Type getVoidPtrType() const {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&lowerTy().getContext(), 8));
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

  /// Construct code sequence to extract the specific value from a `fir.box`.
  mlir::Value getValueFromBox(mlir::Location loc, mlir::Type boxTy,
                              mlir::Value box, mlir::Type resultTy,
                              mlir::ConversionPatternRewriter &rewriter,
                              int boxValue) const {
    if (box.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      auto pty = mlir::LLVM::LLVMPointerType::get(resultTy);
      auto p = rewriter.create<mlir::LLVM::GEPOp>(
          loc, pty, box, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, boxValue});
      auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, resultTy, p);
      attachTBAATag(loadOp, boxTy, nullptr, p);
      return loadOp;
    }
    return rewriter.create<mlir::LLVM::ExtractValueOp>(loc, box, boxValue);
  }

  /// Method to construct code sequence to get the triple for dimension `dim`
  /// from a box.
  llvm::SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys,
                 mlir::Type boxTy, mlir::Value box, mlir::Value dim,
                 mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Value l0 =
        loadDimFieldFromBox(loc, boxTy, box, dim, 0, retTys[0], rewriter);
    mlir::Value l1 =
        loadDimFieldFromBox(loc, boxTy, box, dim, 1, retTys[1], rewriter);
    mlir::Value l2 =
        loadDimFieldFromBox(loc, boxTy, box, dim, 2, retTys[2], rewriter);
    return {l0, l1, l2};
  }

  llvm::SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys,
                 mlir::Type boxTy, mlir::Value box, int dim,
                 mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Value l0 =
        getDimFieldFromBox(loc, boxTy, box, dim, 0, retTys[0], rewriter);
    mlir::Value l1 =
        getDimFieldFromBox(loc, boxTy, box, dim, 1, retTys[1], rewriter);
    mlir::Value l2 =
        getDimFieldFromBox(loc, boxTy, box, dim, 2, retTys[2], rewriter);
    return {l0, l1, l2};
  }

  mlir::Value
  loadDimFieldFromBox(mlir::Location loc, mlir::Type boxTy, mlir::Value box,
                      mlir::Value dim, int off, mlir::Type ty,
                      mlir::ConversionPatternRewriter &rewriter) const {
    assert(box.getType().isa<mlir::LLVM::LLVMPointerType>() &&
           "descriptor inquiry with runtime dim can only be done on descriptor "
           "in memory");
    auto pty = mlir::LLVM::LLVMPointerType::get(ty);
    mlir::LLVM::GEPOp p = genGEP(loc, pty, rewriter, box, 0,
                                 static_cast<int>(kDimsPosInBox), dim, off);
    auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    attachTBAATag(loadOp, boxTy, nullptr, p);
    return loadOp;
  }

  mlir::Value
  getDimFieldFromBox(mlir::Location loc, mlir::Type boxTy, mlir::Value box,
                     int dim, int off, mlir::Type ty,
                     mlir::ConversionPatternRewriter &rewriter) const {
    if (box.getType().isa<mlir::LLVM::LLVMPointerType>()) {
      auto pty = mlir::LLVM::LLVMPointerType::get(ty);
      mlir::LLVM::GEPOp p = genGEP(loc, pty, rewriter, box, 0,
                                   static_cast<int>(kDimsPosInBox), dim, off);
      auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
      attachTBAATag(loadOp, boxTy, nullptr, p);
      return loadOp;
    }
    return rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, box, llvm::ArrayRef<std::int64_t>{kDimsPosInBox, dim, off});
  }

  mlir::Value
  getStrideFromBox(mlir::Location loc, mlir::Type boxTy, mlir::Value box,
                   unsigned dim,
                   mlir::ConversionPatternRewriter &rewriter) const {
    auto idxTy = lowerTy().indexType();
    return getDimFieldFromBox(loc, boxTy, box, dim, kDimStridePos, idxTy,
                              rewriter);
  }

  /// Read base address from a fir.box. Returned address has type ty.
  mlir::Value
  getBaseAddrFromBox(mlir::Location loc, mlir::Type resultTy, mlir::Type boxTy,
                     mlir::Value box,
                     mlir::ConversionPatternRewriter &rewriter) const {
    return getValueFromBox(loc, boxTy, box, resultTy, rewriter, kAddrPosInBox);
  }

  mlir::Value
  getElementSizeFromBox(mlir::Location loc, mlir::Type resultTy,
                        mlir::Type boxTy, mlir::Value box,
                        mlir::ConversionPatternRewriter &rewriter) const {
    return getValueFromBox(loc, boxTy, box, resultTy, rewriter,
                           kElemLenPosInBox);
  }

  // Get the element type given an LLVM type that is of the form
  // [llvm.ptr](array|struct|vector)+ and the provided indexes.
  static mlir::Type getBoxEleTy(mlir::Type type,
                                llvm::ArrayRef<std::int64_t> indexes) {
    if (auto t = type.dyn_cast<mlir::LLVM::LLVMPointerType>())
      type = t.getElementType();
    for (unsigned i : indexes) {
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

  // Return LLVM type of the base address given the LLVM type
  // of the related descriptor (lowered fir.box type).
  static mlir::Type getBaseAddrTypeFromBox(mlir::Type type) {
    return getBoxEleTy(type, {kAddrPosInBox});
  }

  /// Read the address of the type descriptor from a box.
  mlir::Value
  loadTypeDescAddress(mlir::Location loc, mlir::Type boxTy, mlir::Value box,
                      mlir::ConversionPatternRewriter &rewriter) const {
    unsigned typeDescFieldId = getTypeDescFieldId(boxTy);
    mlir::Type tdescType = lowerTy().convertTypeDescType(rewriter.getContext());
    return getValueFromBox(loc, boxTy, box, tdescType, rewriter,
                           typeDescFieldId);
  }

  // Load the attribute from the \p box and perform a check against \p maskValue
  // The final comparison is implemented as `(attribute & maskValue) != 0`.
  mlir::Value genBoxAttributeCheck(mlir::Location loc, mlir::Type boxTy,
                                   mlir::Value box,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   unsigned maskValue) const {
    mlir::Type attrTy = rewriter.getI32Type();
    mlir::Value attribute =
        getValueFromBox(loc, boxTy, box, attrTy, rewriter, kAttributePosInBox);
    mlir::LLVM::ConstantOp attrMask =
        genConstantOffset(loc, rewriter, maskValue);
    auto maskRes =
        rewriter.create<mlir::LLVM::AndOp>(loc, attrTy, attribute, attrMask);
    mlir::LLVM::ConstantOp c0 = genConstantOffset(loc, rewriter, 0);
    return rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, maskRes, c0);
  }

  template <typename... ARGS>
  mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::Type ty,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value base, ARGS... args) const {
    llvm::SmallVector<mlir::LLVM::GEPArg> cv = {args...};
    return rewriter.create<mlir::LLVM::GEPOp>(loc, ty, base, cv);
  }

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
    auto size = genI32Constant(loc, rewriter, 1);
    auto al = rewriter.create<mlir::LLVM::AllocaOp>(loc, toTy, size, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return al;
  }

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }

  void attachTBAATag(mlir::LLVM::AliasAnalysisOpInterface op,
                     mlir::Type baseFIRType, mlir::Type accessFIRType,
                     mlir::LLVM::GEPOp gep) const {
    lowerTy().attachTBAATag(op, baseFIRType, accessFIRType, gep);
  }

  const fir::FIRToLLVMPassOptions &options;
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
} // namespace

namespace {
/// Lower `fir.address_of` operation to `llvm.address_of` operation.
struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AddrOfOp addr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(addr.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
        addr, ty, addr.getSymbol().getRootReference().getValue());
    return mlir::success();
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
  if (auto memSizeFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return memSizeFunc;
  TODO(op.getLoc(), "did not find allocation function");
}

// Compute the alloc scale size (constant factors encoded in the array type).
// We do this for arrays without a constant interior or arrays of character with
// dynamic length arrays, since those are the only ones that get decayed to a
// pointer to the element type.
template <typename OP>
static mlir::Value
genAllocationScaleSize(OP op, mlir::Type ity,
                       mlir::ConversionPatternRewriter &rewriter) {
  mlir::Location loc = op.getLoc();
  mlir::Type dataTy = op.getInType();
  auto seqTy = dataTy.dyn_cast<fir::SequenceType>();
  fir::SequenceType::Extent constSize = 1;
  if (seqTy) {
    int constRows = seqTy.getConstantRows();
    const fir::SequenceType::ShapeRef &shape = seqTy.getShape();
    if (constRows != static_cast<int>(shape.size())) {
      for (auto extent : shape) {
        if (constRows-- > 0)
          continue;
        if (extent != fir::SequenceType::getUnknownExtent())
          constSize *= extent;
      }
    }
  }

  if (constSize != 1) {
    mlir::Value constVal{
        genConstantIndex(loc, ity, rewriter, constSize).getResult()};
    return constVal;
  }
  return nullptr;
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
        size = call.getResult();
        ty = ::getVoidPtrType(alloc.getContext());
      } else {
        return emitError(loc, "unexpected type ")
               << scalarType << " with type parameters";
      }
    }
    if (auto scaleSize = genAllocationScaleSize(alloc, ity, rewriter))
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, scaleSize);
    if (alloc.hasShapeOperands()) {
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
    return mlir::success();
  }
};
} // namespace

namespace {
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
    if (auto argty = boxaddr.getVal().getType().dyn_cast<fir::BaseBoxType>()) {
      rewriter.replaceOp(boxaddr,
                         getBaseAddrFromBox(loc, ty, argty, a, rewriter));
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxaddr, a, 0);
    }
    return mlir::success();
  }
};

/// Convert `!fir.boxchar_len` to  `!llvm.extractvalue` for the 2nd part of the
/// boxchar.
struct BoxCharLenOpConversion : public FIROpConversion<fir::BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxCharLenOp boxCharLen, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value boxChar = adaptor.getOperands()[0];
    mlir::Location loc = boxChar.getLoc();
    mlir::Type returnValTy = boxCharLen.getResult().getType();

    constexpr int boxcharLenIdx = 1;
    auto len = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, boxChar,
                                                           boxcharLenIdx);
    mlir::Value lenAfterCast = integerCast(loc, rewriter, returnValTy, len);
    rewriter.replaceOp(boxCharLen, lenAfterCast);

    return mlir::success();
  }
};

/// Lower `fir.box_dims` to a sequence of operations to extract the requested
/// dimension information from the boxed value.
/// Result in a triple set of GEPs and loads.
struct BoxDimsOpConversion : public FIROpConversion<fir::BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxDimsOp boxdims, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 3> resultTypes = {
        convertType(boxdims.getResult(0).getType()),
        convertType(boxdims.getResult(1).getType()),
        convertType(boxdims.getResult(2).getType()),
    };
    auto results = getDimsFromBox(
        boxdims.getLoc(), resultTypes, boxdims.getVal().getType(),
        adaptor.getOperands()[0], adaptor.getOperands()[1], rewriter);
    rewriter.replaceOp(boxdims, results);
    return mlir::success();
  }
};

/// Lower `fir.box_elesize` to a sequence of operations ro extract the size of
/// an element in the boxed value.
struct BoxEleSizeOpConversion : public FIROpConversion<fir::BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxEleSizeOp boxelesz, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxelesz.getLoc();
    auto ty = convertType(boxelesz.getType());
    auto elemSize = getElementSizeFromBox(loc, ty, boxelesz.getVal().getType(),
                                          box, rewriter);
    rewriter.replaceOp(boxelesz, elemSize);
    return mlir::success();
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
    mlir::Value check = genBoxAttributeCheck(loc, boxisalloc.getVal().getType(),
                                             box, rewriter, kAttrAllocatable);
    rewriter.replaceOp(boxisalloc, check);
    return mlir::success();
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
    auto rank = getValueFromBox(loc, boxisarray.getVal().getType(), a,
                                rewriter.getI32Type(), rewriter, kRankPosInBox);
    auto c0 = genConstantOffset(loc, rewriter, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisarray, mlir::LLVM::ICmpPredicate::ne, rank, c0);
    return mlir::success();
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
    mlir::Value check = genBoxAttributeCheck(loc, boxisptr.getVal().getType(),
                                             box, rewriter, kAttrPointer);
    rewriter.replaceOp(boxisptr, check);
    return mlir::success();
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
    auto result = getValueFromBox(loc, boxrank.getVal().getType(), a, ty,
                                  rewriter, kRankPosInBox);
    rewriter.replaceOp(boxrank, result);
    return mlir::success();
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
    TODO(boxprochost.getLoc(), "fir.boxproc_host codegen");
    return mlir::failure();
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
    auto typeDescAddr = loadTypeDescAddress(
        boxtypedesc.getLoc(), boxtypedesc.getBox().getType(), box, rewriter);
    rewriter.replaceOp(boxtypedesc, typeDescAddr);
    return mlir::success();
  }
};

/// Lower `fir.box_typecode` to a sequence of operations to extract the type
/// code in the boxed value.
struct BoxTypeCodeOpConversion : public FIROpConversion<fir::BoxTypeCodeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::BoxTypeCodeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = box.getLoc();
    auto ty = convertType(op.getType());
    auto typeCode = getValueFromBox(loc, op.getBox().getType(), box, ty,
                                    rewriter, kTypePosInBox);
    rewriter.replaceOp(op, typeCode);
    return mlir::success();
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
      return mlir::success();
    }

    auto charTy = constop.getType().cast<fir::CharacterType>();
    unsigned bits = lowerTy().characterBitsize(charTy);
    mlir::Type intTy = rewriter.getIntegerType(bits);
    mlir::Location loc = constop.getLoc();
    mlir::Value cst = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    if (auto arr = attr.dyn_cast<mlir::DenseElementsAttr>()) {
      cst = rewriter.create<mlir::LLVM::ConstantOp>(loc, ty, arr);
    } else if (auto arr = attr.dyn_cast<mlir::ArrayAttr>()) {
      for (auto a : llvm::enumerate(arr.getValue())) {
        // convert each character to a precise bitsize
        auto elemAttr = mlir::IntegerAttr::get(
            intTy,
            a.value().cast<mlir::IntegerAttr>().getValue().zextOrTrunc(bits));
        auto elemCst =
            rewriter.create<mlir::LLVM::ConstantOp>(loc, intTy, elemAttr);
        cst = rewriter.create<mlir::LLVM::InsertValueOp>(loc, cst, elemCst,
                                                         a.index());
      }
    } else {
      return mlir::failure();
    }
    rewriter.replaceOp(constop, cst);
    return mlir::success();
  }
};

/// `fir.call` -> `llvm.call`
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::CallOp call, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    // Convert arith::FastMathFlagsAttr to LLVM::FastMathFlagsAttr.
    mlir::arith::AttrConvertFastMathToLLVM<fir::CallOp, mlir::LLVM::CallOp>
        attrConvert(call);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        call, resultTys, adaptor.getOperands(), attrConvert.getAttrs());
    return mlir::success();
  }
};
} // namespace

static mlir::Type getComplexEleTy(mlir::Type complex) {
  if (auto cc = complex.dyn_cast<mlir::ComplexType>())
    return cc.getElementType();
  return complex.cast<fir::ComplexType>().getElementType();
}

namespace {
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
    mlir::Type resTy = convertType(cmp.getType());
    mlir::Location loc = cmp.getLoc();
    llvm::SmallVector<mlir::Value, 2> rp = {
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[0], 0),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[1], 0)};
    auto rcp =
        rewriter.create<mlir::LLVM::FCmpOp>(loc, resTy, rp, cmp->getAttrs());
    llvm::SmallVector<mlir::Value, 2> ip = {
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[0], 1),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[1], 1)};
    auto icp =
        rewriter.create<mlir::LLVM::FCmpOp>(loc, resTy, ip, cmp->getAttrs());
    llvm::SmallVector<mlir::Value, 2> cp = {rcp, icp};
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
    return mlir::success();
  }
};

/// Lower complex constants
struct ConstcOpConversion : public FIROpConversion<fir::ConstcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ConstcOp conc, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = conc.getLoc();
    mlir::Type ty = convertType(conc.getType());
    mlir::Type ety = convertType(getComplexEleTy(conc.getType()));
    auto realPart = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, ety, getValue(conc.getReal()));
    auto imPart = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, ety, getValue(conc.getImaginary()));
    auto undef = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto setReal =
        rewriter.create<mlir::LLVM::InsertValueOp>(loc, undef, realPart, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(conc, setReal,
                                                           imPart, 1);
    return mlir::success();
  }

  inline llvm::APFloat getValue(mlir::Attribute attr) const {
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
    auto fromFirTy = convert.getValue().getType();
    auto toFirTy = convert.getRes().getType();
    auto fromTy = convertType(fromFirTy);
    auto toTy = convertType(toFirTy);
    mlir::Value op0 = adaptor.getOperands()[0];

    if (fromFirTy == toFirTy) {
      rewriter.replaceOp(convert, op0);
      return mlir::success();
    }

    auto loc = convert.getLoc();
    auto i1Type = mlir::IntegerType::get(convert.getContext(), 1);

    if (fromFirTy.isa<fir::LogicalType>() || toFirTy.isa<fir::LogicalType>()) {
      // By specification fir::LogicalType value may be any number,
      // where non-zero value represents .true. and zero value represents
      // .false.
      //
      // integer<->logical conversion requires value normalization.
      // Conversion from wide logical to narrow logical must set the result
      // to non-zero iff the input is non-zero - the easiest way to implement
      // it is to compare the input agains zero and set the result to
      // the canonical 0/1.
      // Conversion from narrow logical to wide logical may be implemented
      // as a zero or sign extension of the input, but it may use value
      // normalization as well.
      if (!fromTy.isa<mlir::IntegerType>() || !toTy.isa<mlir::IntegerType>())
        return mlir::emitError(loc)
               << "unsupported types for logical conversion: " << fromTy
               << " -> " << toTy;

      // Do folding for constant inputs.
      if (auto constVal = getIfConstantIntValue(op0)) {
        mlir::Value normVal =
            genConstantIndex(loc, toTy, rewriter, *constVal ? 1 : 0);
        rewriter.replaceOp(convert, normVal);
        return mlir::success();
      }

      // If the input is i1, then we can just zero extend it, and
      // the result will be normalized.
      if (fromTy == i1Type) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(convert, toTy, op0);
        return mlir::success();
      }

      // Compare the input with zero.
      mlir::Value zero = genConstantIndex(loc, fromTy, rewriter, 0);
      auto isTrue = rewriter.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, op0, zero);

      // Zero extend the i1 isTrue result to the required type (unless it is i1
      // itself).
      if (toTy != i1Type)
        rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(convert, toTy, isTrue);
      else
        rewriter.replaceOp(convert, isTrue.getResult());

      return mlir::success();
    }

    if (fromTy == toTy) {
      rewriter.replaceOp(convert, op0);
      return mlir::success();
    }
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
    if (fir::isa_complex(fromFirTy) && fir::isa_complex(toFirTy)) {
      // Special case: handle the conversion of a complex such that both the
      // real and imaginary parts are converted together.
      auto ty = convertType(getComplexEleTy(convert.getValue().getType()));
      auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, op0, 0);
      auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, op0, 1);
      auto nt = convertType(getComplexEleTy(convert.getRes().getType()));
      auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(ty);
      auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(nt);
      auto rc = convertFpToFp(rp, fromBits, toBits, nt);
      auto ic = convertFpToFp(ip, fromBits, toBits, nt);
      auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, toTy);
      auto i1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, un, rc, 0);
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(convert, i1, ic,
                                                             1);
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
        if (fromFirTy == i1Type) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(convert, toTy, op0);
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

/// `fir.disptach_table` operation has no specific CodeGen. The operation is
/// only used to carry information during FIR to FIR passes.
struct DispatchTableOpConversion
    : public FIROpConversion<fir::DispatchTableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DispatchTableOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// `fir.dt_entry` operation has no specific CodeGen. The operation is only used
/// to carry information during FIR to FIR passes.
struct DTEntryOpConversion : public FIROpConversion<fir::DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::DTEntryOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Lower `fir.global_len` operation.
struct GlobalLenOpConversion : public FIROpConversion<fir::GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalLenOp globalLen, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(globalLen.getLoc(), "fir.global_len codegen");
    return mlir::failure();
  }
};

/// Lower fir.len_param_index
struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  mlir::LogicalResult
  matchAndRewrite(fir::LenParamIndexOp lenp, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(lenp.getLoc(), "fir.len_param_index codegen");
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

    mlir::Value charBuffer = operands[0];
    mlir::Value charBufferLen = operands[1];

    mlir::Location loc = emboxChar.getLoc();
    mlir::Type llvmStructTy = convertType(emboxChar.getType());
    auto llvmStruct = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmStructTy);

    mlir::Type lenTy =
        llvmStructTy.cast<mlir::LLVM::LLVMStructType>().getBody()[1];
    mlir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, charBufferLen);

    mlir::Type addrTy =
        llvmStructTy.cast<mlir::LLVM::LLVMStructType>().getBody()[0];
    if (addrTy != charBuffer.getType())
      charBuffer =
          rewriter.create<mlir::LLVM::BitcastOp>(loc, addrTy, charBuffer);

    auto insertBufferOp = rewriter.create<mlir::LLVM::InsertValueOp>(
        loc, llvmStruct, charBuffer, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        emboxChar, insertBufferOp, lenAfterCast, 1);

    return mlir::success();
  }
};
} // namespace

/// Return the LLVMFuncOp corresponding to the standard malloc call.
static mlir::LLVM::LLVMFuncOp
getMalloc(fir::AllocMemOp op, mlir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (mlir::LLVM::LLVMFuncOp mallocFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc"))
    return mallocFunc;
  mlir::OpBuilder moduleBuilder(
      op->getParentOfType<mlir::ModuleOp>().getBodyRegion());
  auto indexType = mlir::IntegerType::get(op.getContext(), 64);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "malloc",
      mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(op.getContext()),
                                        indexType,
                                        /*isVarArg=*/false));
}

/// Helper function for generating the LLVM IR that computes the distance
/// in bytes between adjacent elements pointed to by a pointer
/// of type \p ptrTy. The result is returned as a value of \p idxTy integer
/// type.
static mlir::Value
computeElementDistance(mlir::Location loc, mlir::Type ptrTy, mlir::Type idxTy,
                       mlir::ConversionPatternRewriter &rewriter) {
  // Note that we cannot use something like
  // mlir::LLVM::getPrimitiveTypeSizeInBits() for the element type here. For
  // example, it returns 10 bytes for mlir::Float80Type for targets where it
  // occupies 16 bytes. Proper solution is probably to use
  // mlir::DataLayout::getTypeABIAlignment(), but DataLayout is not being set
  // yet (see llvm-project#57230). For the time being use the '(intptr_t)((type
  // *)0 + 1)' trick for all types. The generated instructions are optimized
  // into constant by the first pass of InstCombine, so it should not be a
  // performance issue.
  auto nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, ptrTy);
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(
      loc, ptrTy, nullPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{1});
  return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, idxTy, gep);
}

/// Return value of the stride in bytes between adjacent elements
/// of LLVM type \p llTy. The result is returned as a value of
/// \p idxTy integer type.
static mlir::Value
genTypeStrideInBytes(mlir::Location loc, mlir::Type idxTy,
                     mlir::ConversionPatternRewriter &rewriter,
                     mlir::Type llTy) {
  // Create a pointer type and use computeElementDistance().
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(llTy);
  return computeElementDistance(loc, ptrTy, idxTy, rewriter);
}

namespace {
/// Lower a `fir.allocmem` instruction into `llvm.call @malloc`
struct AllocMemOpConversion : public FIROpConversion<fir::AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::AllocMemOp heap, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type heapTy = heap.getType();
    mlir::Type ty = convertType(heapTy);
    mlir::LLVM::LLVMFuncOp mallocFunc = getMalloc(heap, rewriter);
    mlir::Location loc = heap.getLoc();
    auto ity = lowerTy().indexType();
    mlir::Type dataTy = fir::unwrapRefType(heapTy);
    if (fir::isRecordWithTypeParameters(fir::unwrapSequenceType(dataTy)))
      TODO(loc, "fir.allocmem codegen of derived type with length parameters");
    mlir::Value size = genTypeSizeInBytes(loc, ity, rewriter, ty);
    if (auto scaleSize = genAllocationScaleSize(heap, ity, rewriter))
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, scaleSize);
    for (mlir::Value opnd : adaptor.getOperands())
      size = rewriter.create<mlir::LLVM::MulOp>(
          loc, ity, size, integerCast(loc, rewriter, ity, opnd));
    heap->setAttr("callee", mlir::SymbolRefAttr::get(mallocFunc));
    auto malloc = rewriter.create<mlir::LLVM::CallOp>(
        loc, ::getVoidPtrType(heap.getContext()), size, heap->getAttrs());
    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(heap, ty,
                                                       malloc.getResult());
    return mlir::success();
  }

  /// Compute the allocation size in bytes of the element type of
  /// \p llTy pointer type. The result is returned as a value of \p idxTy
  /// integer type.
  mlir::Value genTypeSizeInBytes(mlir::Location loc, mlir::Type idxTy,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Type llTy) const {
    auto ptrTy = llTy.dyn_cast<mlir::LLVM::LLVMPointerType>();
    return computeElementDistance(loc, ptrTy, idxTy, rewriter);
  }
};
} // namespace

/// Return the LLVMFuncOp corresponding to the standard free call.
static mlir::LLVM::LLVMFuncOp
getFree(fir::FreeMemOp op, mlir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<mlir::ModuleOp>();
  if (mlir::LLVM::LLVMFuncOp freeFunc =
          module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("free"))
    return freeFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto voidType = mlir::LLVM::LLVMVoidType::get(op.getContext());
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "free",
      mlir::LLVM::LLVMFunctionType::get(voidType,
                                        getVoidPtrType(op.getContext()),
                                        /*isVarArg=*/false));
}

static unsigned getDimension(mlir::LLVM::LLVMArrayType ty) {
  unsigned result = 1;
  for (auto eleTy = ty.getElementType().dyn_cast<mlir::LLVM::LLVMArrayType>();
       eleTy;
       eleTy = eleTy.getElementType().dyn_cast<mlir::LLVM::LLVMArrayType>())
    ++result;
  return result;
}

namespace {
/// Lower a `fir.freemem` instruction into `llvm.call @free`
struct FreeMemOpConversion : public FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FreeMemOp freemem, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::LLVM::LLVMFuncOp freeFunc = getFree(freemem, rewriter);
    mlir::Location loc = freemem.getLoc();
    auto bitcast = rewriter.create<mlir::LLVM::BitcastOp>(
        freemem.getLoc(), voidPtrTy(), adaptor.getOperands()[0]);
    freemem->setAttr("callee", mlir::SymbolRefAttr::get(freeFunc));
    rewriter.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, mlir::ValueRange{bitcast}, freemem->getAttrs());
    rewriter.eraseOp(freemem);
    return mlir::success();
  }
};
} // namespace

// Convert subcomponent array indices from column-major to row-major ordering.
static llvm::SmallVector<mlir::Value>
convertSubcomponentIndices(mlir::Location loc, mlir::Type eleTy,
                           mlir::ValueRange indices,
                           mlir::Type *retTy = nullptr) {
  llvm::SmallVector<mlir::Value> result;
  llvm::SmallVector<mlir::Value> arrayIndices;

  auto appendArrayIndices = [&] {
    if (arrayIndices.empty())
      return;
    std::reverse(arrayIndices.begin(), arrayIndices.end());
    result.append(arrayIndices.begin(), arrayIndices.end());
    arrayIndices.clear();
  };

  for (mlir::Value index : indices) {
    // Component indices can be field index to select a component, or array
    // index, to select an element in an array component.
    if (auto structTy = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(eleTy)) {
      std::int64_t cstIndex = getConstantIntValue(index);
      assert(cstIndex < (int64_t)structTy.getBody().size() &&
             "out-of-bounds struct field index");
      eleTy = structTy.getBody()[cstIndex];
      appendArrayIndices();
      result.push_back(index);
    } else if (auto arrayTy =
                   mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(eleTy)) {
      eleTy = arrayTy.getElementType();
      arrayIndices.push_back(index);
    } else
      fir::emitFatalError(loc, "Unexpected subcomponent type");
  }
  appendArrayIndices();
  if (retTy)
    *retTy = eleTy;
  return result;
}

/// Common base class for embox to descriptor conversion.
template <typename OP>
struct EmboxCommonConversion : public FIROpConversion<OP> {
  using FIROpConversion<OP>::FIROpConversion;

  static int getCFIAttr(fir::BaseBoxType boxTy) {
    auto eleTy = boxTy.getEleTy();
    if (eleTy.isa<fir::PointerType>())
      return CFI_attribute_pointer;
    if (eleTy.isa<fir::HeapType>())
      return CFI_attribute_allocatable;
    return CFI_attribute_other;
  }

  mlir::Value getCharacterByteSize(mlir::Location loc,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   fir::CharacterType charTy,
                                   mlir::ValueRange lenParams) const {
    auto i64Ty = mlir::IntegerType::get(rewriter.getContext(), 64);
    mlir::Value size =
        genTypeStrideInBytes(loc, i64Ty, rewriter, this->convertType(charTy));
    if (charTy.hasConstantLen())
      return size; // Length accounted for in the genTypeStrideInBytes GEP.
    // Otherwise,  multiply the single character size by the length.
    assert(!lenParams.empty());
    auto len64 = FIROpConversion<OP>::integerCast(loc, rewriter, i64Ty,
                                                  lenParams.back());
    return rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, size, len64);
  }

  // Get the element size and CFI type code of the boxed value.
  std::tuple<mlir::Value, mlir::Value> getSizeAndTypeCode(
      mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
      mlir::Type boxEleTy, mlir::ValueRange lenParams = {}) const {
    auto i64Ty = mlir::IntegerType::get(rewriter.getContext(), 64);
    if (auto eleTy = fir::dyn_cast_ptrEleTy(boxEleTy))
      boxEleTy = eleTy;
    if (auto seqTy = boxEleTy.dyn_cast<fir::SequenceType>())
      return getSizeAndTypeCode(loc, rewriter, seqTy.getEleTy(), lenParams);
    if (boxEleTy.isa<mlir::NoneType>()) // unlimited polymorphic or assumed type
      return {rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, 0),
              this->genConstantOffset(loc, rewriter, CFI_type_other)};
    mlir::Value typeCodeVal = this->genConstantOffset(
        loc, rewriter,
        fir::getTypeCode(boxEleTy, this->lowerTy().getKindMap()));
    if (fir::isa_integer(boxEleTy) || boxEleTy.dyn_cast<fir::LogicalType>() ||
        fir::isa_real(boxEleTy) || fir::isa_complex(boxEleTy))
      return {genTypeStrideInBytes(loc, i64Ty, rewriter,
                                   this->convertType(boxEleTy)),
              typeCodeVal};
    if (auto charTy = boxEleTy.dyn_cast<fir::CharacterType>())
      return {getCharacterByteSize(loc, rewriter, charTy, lenParams),
              typeCodeVal};
    if (fir::isa_ref_type(boxEleTy)) {
      auto ptrTy = mlir::LLVM::LLVMPointerType::get(
          mlir::LLVM::LLVMVoidType::get(rewriter.getContext()));
      return {genTypeStrideInBytes(loc, i64Ty, rewriter, ptrTy), typeCodeVal};
    }
    if (boxEleTy.isa<fir::RecordType>())
      return {genTypeStrideInBytes(loc, i64Ty, rewriter,
                                   this->convertType(boxEleTy)),
              typeCodeVal};
    fir::emitFatalError(loc, "unhandled type in fir.box code generation");
  }

  /// Basic pattern to write a field in the descriptor
  mlir::Value insertField(mlir::ConversionPatternRewriter &rewriter,
                          mlir::Location loc, mlir::Value dest,
                          llvm::ArrayRef<std::int64_t> fldIndexes,
                          mlir::Value value, bool bitcast = false) const {
    auto boxTy = dest.getType();
    auto fldTy = this->getBoxEleTy(boxTy, fldIndexes);
    if (bitcast)
      value = rewriter.create<mlir::LLVM::BitcastOp>(loc, fldTy, value);
    else
      value = this->integerCast(loc, rewriter, fldTy, value);
    return rewriter.create<mlir::LLVM::InsertValueOp>(loc, dest, value,
                                                      fldIndexes);
  }

  inline mlir::Value
  insertBaseAddress(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, mlir::Value dest,
                    mlir::Value base) const {
    return insertField(rewriter, loc, dest, {kAddrPosInBox}, base,
                       /*bitCast=*/true);
  }

  inline mlir::Value insertLowerBound(mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Location loc, mlir::Value dest,
                                      unsigned dim, mlir::Value lb) const {
    return insertField(rewriter, loc, dest,
                       {kDimsPosInBox, dim, kDimLowerBoundPos}, lb);
  }

  inline mlir::Value insertExtent(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value dest,
                                  unsigned dim, mlir::Value extent) const {
    return insertField(rewriter, loc, dest, {kDimsPosInBox, dim, kDimExtentPos},
                       extent);
  }

  inline mlir::Value insertStride(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value dest,
                                  unsigned dim, mlir::Value stride) const {
    return insertField(rewriter, loc, dest, {kDimsPosInBox, dim, kDimStridePos},
                       stride);
  }

  /// Get the address of the type descriptor global variable that was created by
  /// lowering for derived type \p recType.
  mlir::Value getTypeDescriptor(mlir::ModuleOp mod,
                                mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc,
                                fir::RecordType recType) const {
    std::string name =
        fir::NameUniquer::getTypeDescriptorName(recType.getName());
    if (auto global = mod.template lookupSymbol<fir::GlobalOp>(name)) {
      auto ty = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(global.getType()));
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty,
                                                      global.getSymName());
    }
    if (auto global = mod.template lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
      // The global may have already been translated to LLVM.
      auto ty = mlir::LLVM::LLVMPointerType::get(global.getType());
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ty,
                                                      global.getSymName());
    }
    // Type info derived types do not have type descriptors since they are the
    // types defining type descriptors.
    if (!this->options.ignoreMissingTypeDescriptors &&
        !fir::NameUniquer::belongsToModule(
            name, Fortran::semantics::typeInfoBuiltinModule))
      fir::emitFatalError(
          loc, "runtime derived type info descriptor was not generated");
    return rewriter.create<mlir::LLVM::NullOp>(
        loc, ::getVoidPtrType(mod.getContext()));
  }

  mlir::Value populateDescriptor(mlir::Location loc, mlir::ModuleOp mod,
                                 fir::BaseBoxType boxTy, mlir::Type inputType,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 unsigned rank, mlir::Value eleSize,
                                 mlir::Value cfiTy,
                                 mlir::Value typeDesc) const {
    auto convTy = this->lowerTy().convertBoxType(boxTy, rank);
    auto llvmBoxPtrTy = convTy.template cast<mlir::LLVM::LLVMPointerType>();
    auto llvmBoxTy = llvmBoxPtrTy.getElementType();
    bool isUnlimitedPolymorphic = fir::isUnlimitedPolymorphicType(boxTy);
    bool useInputType = fir::isPolymorphicType(boxTy) || isUnlimitedPolymorphic;
    mlir::Value descriptor =
        rewriter.create<mlir::LLVM::UndefOp>(loc, llvmBoxTy);
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
    const bool hasAddendum = fir::boxHasAddendum(boxTy);
    descriptor =
        insertField(rewriter, loc, descriptor, {kF18AddendumPosInBox},
                    this->genI32Constant(loc, rewriter, hasAddendum ? 1 : 0));

    if (hasAddendum) {
      unsigned typeDescFieldId = getTypeDescFieldId(boxTy);
      if (!typeDesc) {
        if (useInputType) {
          mlir::Type innerType = fir::unwrapInnerType(inputType);
          if (innerType && innerType.template isa<fir::RecordType>()) {
            auto recTy = innerType.template dyn_cast<fir::RecordType>();
            typeDesc = getTypeDescriptor(mod, rewriter, loc, recTy);
          } else {
            // Unlimited polymorphic type descriptor with no record type. Set
            // type descriptor address to a clean state.
            typeDesc = rewriter.create<mlir::LLVM::NullOp>(
                loc, ::getVoidPtrType(mod.getContext()));
          }
        } else {
          typeDesc = getTypeDescriptor(mod, rewriter, loc,
                                       fir::unwrapIfDerived(boxTy));
        }
      }
      if (typeDesc)
        descriptor =
            insertField(rewriter, loc, descriptor, {typeDescFieldId}, typeDesc,
                        /*bitCast=*/true);
    }
    return descriptor;
  }

  // Template used for fir::EmboxOp and fir::cg::XEmboxOp
  template <typename BOX>
  std::tuple<fir::BaseBoxType, mlir::Value, mlir::Value>
  consDescriptorPrefix(BOX box, mlir::Type inputType,
                       mlir::ConversionPatternRewriter &rewriter, unsigned rank,
                       [[maybe_unused]] mlir::ValueRange substrParams,
                       mlir::ValueRange lenParams, mlir::Value sourceBox = {},
                       mlir::Type sourceBoxType = {}) const {
    auto loc = box.getLoc();
    auto boxTy = box.getType().template dyn_cast<fir::BaseBoxType>();
    bool useInputType = fir::isPolymorphicType(boxTy) &&
                        !fir::isUnlimitedPolymorphicType(inputType);
    llvm::SmallVector<mlir::Value> typeparams = lenParams;
    if constexpr (!std::is_same_v<BOX, fir::EmboxOp>) {
      if (!box.getSubstr().empty() && fir::hasDynamicSize(boxTy.getEleTy()))
        typeparams.push_back(substrParams[1]);
    }

    // Write each of the fields with the appropriate values.
    // When emboxing an element to a polymorphic descriptor, use the
    // input type since the destination descriptor type has not the exact
    // information.
    auto [eleSize, cfiTy] = getSizeAndTypeCode(
        loc, rewriter, useInputType ? inputType : boxTy.getEleTy(), typeparams);

    mlir::Value typeDesc;
    // When emboxing to a polymorphic box, get the type descriptor, type code
    // and element size from the source box if any.
    if (fir::isPolymorphicType(boxTy) && sourceBox) {
      typeDesc =
          this->loadTypeDescAddress(loc, sourceBoxType, sourceBox, rewriter);
      mlir::Type idxTy = this->lowerTy().indexType();
      eleSize = this->getElementSizeFromBox(loc, idxTy, sourceBoxType,
                                            sourceBox, rewriter);
      cfiTy = this->getValueFromBox(loc, sourceBoxType, sourceBox,
                                    cfiTy.getType(), rewriter, kTypePosInBox);
    }
    auto mod = box->template getParentOfType<mlir::ModuleOp>();
    mlir::Value descriptor = populateDescriptor(
        loc, mod, boxTy, inputType, rewriter, rank, eleSize, cfiTy, typeDesc);

    return {boxTy, descriptor, eleSize};
  }

  std::tuple<fir::BaseBoxType, mlir::Value, mlir::Value>
  consDescriptorPrefix(fir::cg::XReboxOp box, mlir::Value loweredBox,
                       mlir::ConversionPatternRewriter &rewriter, unsigned rank,
                       mlir::ValueRange substrParams,
                       mlir::ValueRange lenParams,
                       mlir::Value typeDesc = {}) const {
    auto loc = box.getLoc();
    auto boxTy = box.getType().dyn_cast<fir::BaseBoxType>();
    auto inputBoxTy = box.getBox().getType().dyn_cast<fir::BaseBoxType>();
    llvm::SmallVector<mlir::Value> typeparams = lenParams;
    if (!box.getSubstr().empty() && fir::hasDynamicSize(boxTy.getEleTy()))
      typeparams.push_back(substrParams[1]);

    auto [eleSize, cfiTy] =
        getSizeAndTypeCode(loc, rewriter, boxTy.getEleTy(), typeparams);

    // Reboxing to a polymorphic entity. eleSize and type code need to
    // be retrieved from the initial box and propagated to the new box.
    // If the initial box has an addendum, the type desc must be propagated as
    // well.
    if (fir::isPolymorphicType(boxTy)) {
      mlir::Type idxTy = this->lowerTy().indexType();
      eleSize =
          this->getElementSizeFromBox(loc, idxTy, boxTy, loweredBox, rewriter);
      cfiTy = this->getValueFromBox(loc, boxTy, loweredBox, cfiTy.getType(),
                                    rewriter, kTypePosInBox);
      // TODO: For initial box that are unlimited polymorphic entities, this
      // code must be made conditional because unlimited polymorphic entities
      // with intrinsic type spec does not have addendum.
      if (fir::boxHasAddendum(inputBoxTy))
        typeDesc = this->loadTypeDescAddress(loc, box.getBox().getType(),
                                             loweredBox, rewriter);
    }

    auto mod = box->template getParentOfType<mlir::ModuleOp>();
    mlir::Value descriptor =
        populateDescriptor(loc, mod, boxTy, box.getBox().getType(), rewriter,
                           rank, eleSize, cfiTy, typeDesc);

    return {boxTy, descriptor, eleSize};
  }

  // Compute the base address of a fir.box given the indices from the slice.
  // The indices from the "outer" dimensions (every dimension after the first
  // one (inlcuded) that is not a compile time constant) must have been
  // multiplied with the related extents and added together into \p outerOffset.
  mlir::Value
  genBoxOffsetGep(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  mlir::Value base, mlir::Value outerOffset,
                  mlir::ValueRange cstInteriorIndices,
                  mlir::ValueRange componentIndices,
                  std::optional<mlir::Value> substringOffset) const {
    llvm::SmallVector<mlir::LLVM::GEPArg> gepArgs{outerOffset};
    mlir::Type resultTy =
        base.getType().cast<mlir::LLVM::LLVMPointerType>().getElementType();
    // Fortran is column major, llvm GEP is row major: reverse the indices here.
    for (mlir::Value interiorIndex : llvm::reverse(cstInteriorIndices)) {
      auto arrayTy = resultTy.dyn_cast<mlir::LLVM::LLVMArrayType>();
      if (!arrayTy)
        fir::emitFatalError(
            loc,
            "corrupted GEP generated being generated in fir.embox/fir.rebox");
      resultTy = arrayTy.getElementType();
      gepArgs.push_back(interiorIndex);
    }
    llvm::SmallVector<mlir::Value> gepIndices =
        convertSubcomponentIndices(loc, resultTy, componentIndices, &resultTy);
    gepArgs.append(gepIndices.begin(), gepIndices.end());
    if (substringOffset) {
      if (auto arrayTy = resultTy.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
        gepArgs.push_back(*substringOffset);
        resultTy = arrayTy.getElementType();
      } else {
        // If the CHARACTER length is dynamic, the whole base type should have
        // degenerated to an llvm.ptr<i[width]>, and there should not be any
        // cstInteriorIndices/componentIndices. The substring offset can be
        // added to the outterOffset since it applies on the same LLVM type.
        if (gepArgs.size() != 1)
          fir::emitFatalError(loc,
                              "corrupted substring GEP in fir.embox/fir.rebox");
        mlir::Type outterOffsetTy = gepArgs[0].get<mlir::Value>().getType();
        mlir::Value cast =
            this->integerCast(loc, rewriter, outterOffsetTy, *substringOffset);

        gepArgs[0] = rewriter.create<mlir::LLVM::AddOp>(
            loc, outterOffsetTy, gepArgs[0].get<mlir::Value>(), cast);
      }
    }
    resultTy = mlir::LLVM::LLVMPointerType::get(resultTy);
    return rewriter.create<mlir::LLVM::GEPOp>(loc, resultTy, base, gepArgs);
  }

  template <typename BOX>
  void
  getSubcomponentIndices(BOX xbox, mlir::Value memref,
                         mlir::ValueRange operands,
                         mlir::SmallVectorImpl<mlir::Value> &indices) const {
    // For each field in the path add the offset to base via the args list.
    // In the most general case, some offsets must be computed since
    // they are not be known until runtime.
    if (fir::hasDynamicSize(fir::unwrapSequenceType(
            fir::unwrapPassByRefType(memref.getType()))))
      TODO(xbox.getLoc(),
           "fir.embox codegen dynamic size component in derived type");
    indices.append(operands.begin() + xbox.subcomponentOffset(),
                   operands.begin() + xbox.subcomponentOffset() +
                       xbox.getSubcomponent().size());
  }

  static bool isInGlobalOp(mlir::ConversionPatternRewriter &rewriter) {
    auto *thisBlock = rewriter.getInsertionBlock();
    return thisBlock &&
           mlir::isa<mlir::LLVM::GlobalOp>(thisBlock->getParentOp());
  }

  /// If the embox is not in a globalOp body, allocate storage for the box;
  /// store the value inside and return the generated alloca. Return the input
  /// value otherwise.
  mlir::Value
  placeInMemoryIfNotGlobalInit(mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc, mlir::Type boxTy,
                               mlir::Value boxValue) const {
    if (isInGlobalOp(rewriter))
      return boxValue;
    auto boxPtrTy = mlir::LLVM::LLVMPointerType::get(boxValue.getType());
    auto alloca =
        this->genAllocaWithType(loc, boxPtrTy, defaultAlign, rewriter);
    auto storeOp = rewriter.create<mlir::LLVM::StoreOp>(loc, boxValue, alloca);
    this->attachTBAATag(storeOp, boxTy, boxTy, nullptr);
    return alloca;
  }
};

/// Compute the extent of a triplet slice (lb:ub:step).
static mlir::Value
computeTripletExtent(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, mlir::Value lb, mlir::Value ub,
                     mlir::Value step, mlir::Value zero, mlir::Type type) {
  mlir::Value extent = rewriter.create<mlir::LLVM::SubOp>(loc, type, ub, lb);
  extent = rewriter.create<mlir::LLVM::AddOp>(loc, type, extent, step);
  extent = rewriter.create<mlir::LLVM::SDivOp>(loc, type, extent, step);
  // If the resulting extent is negative (`ub-lb` and `step` have different
  // signs), zero must be returned instead.
  auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
      loc, mlir::LLVM::ICmpPredicate::sgt, extent, zero);
  return rewriter.create<mlir::LLVM::SelectOp>(loc, cmp, extent, zero);
}

/// Create a generic box on a memory reference. This conversions lowers the
/// abstract box to the appropriate, initialized descriptor.
struct EmboxOpConversion : public EmboxCommonConversion<fir::EmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxOp embox, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::Value sourceBox;
    mlir::Type sourceBoxType;
    if (embox.getSourceBox()) {
      sourceBox = operands[embox.getSourceBoxOffset()];
      sourceBoxType = embox.getSourceBox().getType();
    }
    assert(!embox.getShape() && "There should be no dims on this embox op");
    auto [boxTy, dest, eleSize] = consDescriptorPrefix(
        embox, fir::unwrapRefType(embox.getMemref().getType()), rewriter,
        /*rank=*/0, /*substrParams=*/mlir::ValueRange{},
        adaptor.getTypeparams(), sourceBox, sourceBoxType);
    dest = insertBaseAddress(rewriter, embox.getLoc(), dest, operands[0]);
    if (fir::isDerivedTypeWithLenParams(boxTy)) {
      TODO(embox.getLoc(),
           "fir.embox codegen of derived with length parameters");
      return mlir::failure();
    }
    auto result =
        placeInMemoryIfNotGlobalInit(rewriter, embox.getLoc(), boxTy, dest);
    rewriter.replaceOp(embox, result);
    return mlir::success();
  }
};

/// Create a generic box on a memory reference.
struct XEmboxOpConversion : public EmboxCommonConversion<fir::cg::XEmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XEmboxOp xbox, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::Value sourceBox;
    mlir::Type sourceBoxType;
    if (xbox.getSourceBox()) {
      sourceBox = operands[xbox.getSourceBoxOffset()];
      sourceBoxType = xbox.getSourceBox().getType();
    }
    auto [boxTy, dest, resultEleSize] = consDescriptorPrefix(
        xbox, fir::unwrapRefType(xbox.getMemref().getType()), rewriter,
        xbox.getOutRank(), adaptor.getSubstr(), adaptor.getLenParams(),
        sourceBox, sourceBoxType);
    // Generate the triples in the dims field of the descriptor
    auto i64Ty = mlir::IntegerType::get(xbox.getContext(), 64);
    mlir::Value base = operands[0];
    assert(!xbox.getShape().empty() && "must have a shape");
    unsigned shapeOffset = xbox.shapeOffset();
    bool hasShift = !xbox.getShift().empty();
    unsigned shiftOffset = xbox.shiftOffset();
    bool hasSlice = !xbox.getSlice().empty();
    unsigned sliceOffset = xbox.sliceOffset();
    mlir::Location loc = xbox.getLoc();
    mlir::Value zero = genConstantIndex(loc, i64Ty, rewriter, 0);
    mlir::Value one = genConstantIndex(loc, i64Ty, rewriter, 1);
    mlir::Value prevPtrOff = one;
    mlir::Type eleTy = boxTy.getEleTy();
    const unsigned rank = xbox.getRank();
    llvm::SmallVector<mlir::Value> cstInteriorIndices;
    unsigned constRows = 0;
    mlir::Value ptrOffset = zero;
    mlir::Type memEleTy = fir::dyn_cast_ptrEleTy(xbox.getMemref().getType());
    assert(memEleTy.isa<fir::SequenceType>());
    auto seqTy = memEleTy.cast<fir::SequenceType>();
    mlir::Type seqEleTy = seqTy.getEleTy();
    // Adjust the element scaling factor if the element is a dependent type.
    if (fir::hasDynamicSize(seqEleTy)) {
      if (auto charTy = seqEleTy.dyn_cast<fir::CharacterType>()) {
        // The GEP pointer type decays to llvm.ptr<i[width]>.
        // The scaling factor is the runtime value of the length.
        assert(!adaptor.getLenParams().empty());
        prevPtrOff = FIROpConversion::integerCast(
            loc, rewriter, i64Ty, adaptor.getLenParams().back());
      } else if (seqEleTy.isa<fir::RecordType>()) {
        // prevPtrOff = ;
        TODO(loc, "generate call to calculate size of PDT");
      } else {
        fir::emitFatalError(loc, "unexpected dynamic type");
      }
    } else {
      constRows = seqTy.getConstantRows();
    }

    const auto hasSubcomp = !xbox.getSubcomponent().empty();
    const bool hasSubstr = !xbox.getSubstr().empty();
    // Initial element stride that will be use to compute the step in
    // each dimension. Initially, this is the size of the input element.
    // Note that when there are no components/substring, the resultEleSize
    // that was previously computed matches the input element size.
    mlir::Value prevDimByteStride = resultEleSize;
    if (hasSubcomp) {
      // We have a subcomponent. The step value needs to be the number of
      // bytes per element (which is a derived type).
      prevDimByteStride =
          genTypeStrideInBytes(loc, i64Ty, rewriter, convertType(seqEleTy));
    } else if (hasSubstr) {
      // We have a substring. The step value needs to be the number of bytes
      // per CHARACTER element.
      auto charTy = seqEleTy.cast<fir::CharacterType>();
      if (fir::hasDynamicSize(charTy)) {
        prevDimByteStride =
            getCharacterByteSize(loc, rewriter, charTy, adaptor.getLenParams());
      } else {
        prevDimByteStride = genConstantIndex(
            loc, i64Ty, rewriter,
            charTy.getLen() * lowerTy().characterBitsize(charTy) / 8);
      }
    }

    // Process the array subspace arguments (shape, shift, etc.), if any,
    // translating everything to values in the descriptor wherever the entity
    // has a dynamic array dimension.
    for (unsigned di = 0, descIdx = 0; di < rank; ++di) {
      mlir::Value extent = operands[shapeOffset];
      mlir::Value outerExtent = extent;
      bool skipNext = false;
      if (hasSlice) {
        mlir::Value off = operands[sliceOffset];
        mlir::Value adj = one;
        if (hasShift)
          adj = operands[shiftOffset];
        auto ao = rewriter.create<mlir::LLVM::SubOp>(loc, i64Ty, off, adj);
        if (constRows > 0) {
          cstInteriorIndices.push_back(ao);
        } else {
          auto dimOff =
              rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, ao, prevPtrOff);
          ptrOffset =
              rewriter.create<mlir::LLVM::AddOp>(loc, i64Ty, dimOff, ptrOffset);
        }
        if (mlir::isa_and_nonnull<fir::UndefOp>(
                xbox.getSlice()[3 * di + 1].getDefiningOp())) {
          // This dimension contains a scalar expression in the array slice op.
          // The dimension is loop invariant, will be dropped, and will not
          // appear in the descriptor.
          skipNext = true;
        }
      }
      if (!skipNext) {
        // store extent
        if (hasSlice)
          extent = computeTripletExtent(rewriter, loc, operands[sliceOffset],
                                        operands[sliceOffset + 1],
                                        operands[sliceOffset + 2], zero, i64Ty);
        // Lower bound is normalized to 0 for BIND(C) interoperability.
        mlir::Value lb = zero;
        const bool isaPointerOrAllocatable =
            eleTy.isa<fir::PointerType>() || eleTy.isa<fir::HeapType>();
        // Lower bound is defaults to 1 for POINTER, ALLOCATABLE, and
        // denormalized descriptors.
        if (isaPointerOrAllocatable || !normalizedLowerBound(xbox))
          lb = one;
        // If there is a shifted origin, and no fir.slice, and this is not
        // a normalized descriptor then use the value from the shift op as
        // the lower bound.
        if (hasShift && !(hasSlice || hasSubcomp || hasSubstr) &&
            (isaPointerOrAllocatable || !normalizedLowerBound(xbox))) {
          lb = operands[shiftOffset];
          auto extentIsEmpty = rewriter.create<mlir::LLVM::ICmpOp>(
              loc, mlir::LLVM::ICmpPredicate::eq, extent, zero);
          lb = rewriter.create<mlir::LLVM::SelectOp>(loc, extentIsEmpty, one,
                                                     lb);
        }
        dest = insertLowerBound(rewriter, loc, dest, descIdx, lb);

        dest = insertExtent(rewriter, loc, dest, descIdx, extent);

        // store step (scaled by shaped extent)
        mlir::Value step = prevDimByteStride;
        if (hasSlice)
          step = rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, step,
                                                    operands[sliceOffset + 2]);
        dest = insertStride(rewriter, loc, dest, descIdx, step);
        ++descIdx;
      }

      // compute the stride and offset for the next natural dimension
      prevDimByteStride = rewriter.create<mlir::LLVM::MulOp>(
          loc, i64Ty, prevDimByteStride, outerExtent);
      if (constRows == 0)
        prevPtrOff = rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, prevPtrOff,
                                                        outerExtent);
      else
        --constRows;

      // increment iterators
      ++shapeOffset;
      if (hasShift)
        ++shiftOffset;
      if (hasSlice)
        sliceOffset += 3;
    }
    if (hasSlice || hasSubcomp || hasSubstr) {
      // Shift the base address.
      llvm::SmallVector<mlir::Value> fieldIndices;
      std::optional<mlir::Value> substringOffset;
      if (hasSubcomp)
        getSubcomponentIndices(xbox, xbox.getMemref(), operands, fieldIndices);
      if (hasSubstr)
        substringOffset = operands[xbox.substrOffset()];
      base = genBoxOffsetGep(rewriter, loc, base, ptrOffset, cstInteriorIndices,
                             fieldIndices, substringOffset);
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    if (fir::isDerivedTypeWithLenParams(boxTy))
      TODO(loc, "fir.embox codegen of derived with length parameters");

    mlir::Value result =
        placeInMemoryIfNotGlobalInit(rewriter, loc, boxTy, dest);
    rewriter.replaceOp(xbox, result);
    return mlir::success();
  }

  /// Return true if `xbox` has a normalized lower bounds attribute. A box value
  /// that is neither a POINTER nor an ALLOCATABLE should be normalized to a
  /// zero origin lower bound for interoperability with BIND(C).
  inline static bool normalizedLowerBound(fir::cg::XEmboxOp xbox) {
    return xbox->hasAttr(fir::getNormalizedLowerBoundAttrName());
  }
};

/// Create a new box given a box reference.
struct XReboxOpConversion : public EmboxCommonConversion<fir::cg::XReboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::cg::XReboxOp rebox, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = rebox.getLoc();
    mlir::Type idxTy = lowerTy().indexType();
    mlir::Value loweredBox = adaptor.getOperands()[0];
    mlir::ValueRange operands = adaptor.getOperands();

    // Inside a fir.global, the input box was produced as an llvm.struct<>
    // because objects cannot be handled in memory inside a fir.global body that
    // must be constant foldable. However, the type translation are not
    // contextual, so the fir.box<T> type of the operation that produced the
    // fir.box was translated to an llvm.ptr<llvm.struct<>> and the MLIR pass
    // manager inserted a builtin.unrealized_conversion_cast that was inserted
    // and needs to be removed here.
    if (isInGlobalOp(rewriter))
      if (auto unrealizedCast =
              loweredBox.getDefiningOp<mlir::UnrealizedConversionCastOp>())
        loweredBox = unrealizedCast.getInputs()[0];

    // Create new descriptor and fill its non-shape related data.
    llvm::SmallVector<mlir::Value, 2> lenParams;
    mlir::Type inputEleTy = getInputEleTy(rebox);
    if (auto charTy = inputEleTy.dyn_cast<fir::CharacterType>()) {
      if (charTy.hasConstantLen()) {
        mlir::Value len =
            genConstantIndex(loc, idxTy, rewriter, charTy.getLen());
        lenParams.emplace_back(len);
      } else {
        mlir::Value len = getElementSizeFromBox(
            loc, idxTy, rebox.getBox().getType(), loweredBox, rewriter);
        if (charTy.getFKind() != 1) {
          assert(!isInGlobalOp(rewriter) &&
                 "character target in global op must have constant length");
          mlir::Value width =
              genConstantIndex(loc, idxTy, rewriter, charTy.getFKind());
          len = rewriter.create<mlir::LLVM::SDivOp>(loc, idxTy, len, width);
        }
        lenParams.emplace_back(len);
      }
    } else if (auto recTy = inputEleTy.dyn_cast<fir::RecordType>()) {
      if (recTy.getNumLenParams() != 0)
        TODO(loc, "reboxing descriptor of derived type with length parameters");
    }

    // Rebox on polymorphic entities needs to carry over the dynamic type.
    mlir::Value typeDescAddr;
    if (rebox.getBox().getType().isa<fir::ClassType>() &&
        rebox.getType().isa<fir::ClassType>())
      typeDescAddr = loadTypeDescAddress(loc, rebox.getBox().getType(),
                                         loweredBox, rewriter);

    auto [boxTy, dest, eleSize] =
        consDescriptorPrefix(rebox, loweredBox, rewriter, rebox.getOutRank(),
                             adaptor.getSubstr(), lenParams, typeDescAddr);

    // Read input extents, strides, and base address
    llvm::SmallVector<mlir::Value> inputExtents;
    llvm::SmallVector<mlir::Value> inputStrides;
    const unsigned inputRank = rebox.getRank();
    for (unsigned dim = 0; dim < inputRank; ++dim) {
      llvm::SmallVector<mlir::Value, 3> dimInfo =
          getDimsFromBox(loc, {idxTy, idxTy, idxTy}, rebox.getBox().getType(),
                         loweredBox, dim, rewriter);
      inputExtents.emplace_back(dimInfo[1]);
      inputStrides.emplace_back(dimInfo[2]);
    }

    mlir::Type baseTy = getBaseAddrTypeFromBox(loweredBox.getType());
    mlir::Value baseAddr = getBaseAddrFromBox(
        loc, baseTy, rebox.getBox().getType(), loweredBox, rewriter);

    if (!rebox.getSlice().empty() || !rebox.getSubcomponent().empty())
      return sliceBox(rebox, boxTy, dest, baseAddr, inputExtents, inputStrides,
                      operands, rewriter);
    return reshapeBox(rebox, boxTy, dest, baseAddr, inputExtents, inputStrides,
                      operands, rewriter);
  }

private:
  /// Write resulting shape and base address in descriptor, and replace rebox
  /// op.
  mlir::LogicalResult
  finalizeRebox(fir::cg::XReboxOp rebox, mlir::Type destBoxTy, mlir::Value dest,
                mlir::Value base, mlir::ValueRange lbounds,
                mlir::ValueRange extents, mlir::ValueRange strides,
                mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Location loc = rebox.getLoc();
    mlir::Value zero =
        genConstantIndex(loc, lowerTy().indexType(), rewriter, 0);
    mlir::Value one = genConstantIndex(loc, lowerTy().indexType(), rewriter, 1);
    for (auto iter : llvm::enumerate(llvm::zip(extents, strides))) {
      mlir::Value extent = std::get<0>(iter.value());
      unsigned dim = iter.index();
      mlir::Value lb = one;
      if (!lbounds.empty()) {
        lb = lbounds[dim];
        auto extentIsEmpty = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, extent, zero);
        lb = rewriter.create<mlir::LLVM::SelectOp>(loc, extentIsEmpty, one, lb);
      };
      dest = insertLowerBound(rewriter, loc, dest, dim, lb);
      dest = insertExtent(rewriter, loc, dest, dim, extent);
      dest = insertStride(rewriter, loc, dest, dim, std::get<1>(iter.value()));
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    mlir::Value result =
        placeInMemoryIfNotGlobalInit(rewriter, rebox.getLoc(), destBoxTy, dest);
    rewriter.replaceOp(rebox, result);
    return mlir::success();
  }

  // Apply slice given the base address, extents and strides of the input box.
  mlir::LogicalResult
  sliceBox(fir::cg::XReboxOp rebox, mlir::Type destBoxTy, mlir::Value dest,
           mlir::Value base, mlir::ValueRange inputExtents,
           mlir::ValueRange inputStrides, mlir::ValueRange operands,
           mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Location loc = rebox.getLoc();
    mlir::Type voidPtrTy = ::getVoidPtrType(rebox.getContext());
    mlir::Type idxTy = lowerTy().indexType();
    mlir::Value zero = genConstantIndex(loc, idxTy, rewriter, 0);
    // Apply subcomponent and substring shift on base address.
    if (!rebox.getSubcomponent().empty() || !rebox.getSubstr().empty()) {
      // Cast to inputEleTy* so that a GEP can be used.
      mlir::Type inputEleTy = getInputEleTy(rebox);
      auto llvmElePtrTy =
          mlir::LLVM::LLVMPointerType::get(convertType(inputEleTy));
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, llvmElePtrTy, base);

      llvm::SmallVector<mlir::Value> fieldIndices;
      std::optional<mlir::Value> substringOffset;
      if (!rebox.getSubcomponent().empty())
        getSubcomponentIndices(rebox, rebox.getBox(), operands, fieldIndices);
      if (!rebox.getSubstr().empty())
        substringOffset = operands[rebox.substrOffset()];
      base = genBoxOffsetGep(rewriter, loc, base, zero,
                             /*cstInteriorIndices=*/std::nullopt, fieldIndices,
                             substringOffset);
    }

    if (rebox.getSlice().empty())
      // The array section is of the form array[%component][substring], keep
      // the input array extents and strides.
      return finalizeRebox(rebox, destBoxTy, dest, base,
                           /*lbounds*/ std::nullopt, inputExtents, inputStrides,
                           rewriter);

    // Strides from the fir.box are in bytes.
    base = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, base);

    // The slice is of the form array(i:j:k)[%component]. Compute new extents
    // and strides.
    llvm::SmallVector<mlir::Value> slicedExtents;
    llvm::SmallVector<mlir::Value> slicedStrides;
    mlir::Value one = genConstantIndex(loc, idxTy, rewriter, 1);
    const bool sliceHasOrigins = !rebox.getShift().empty();
    unsigned sliceOps = rebox.sliceOffset();
    unsigned shiftOps = rebox.shiftOffset();
    auto strideOps = inputStrides.begin();
    const unsigned inputRank = inputStrides.size();
    for (unsigned i = 0; i < inputRank;
         ++i, ++strideOps, ++shiftOps, sliceOps += 3) {
      mlir::Value sliceLb =
          integerCast(loc, rewriter, idxTy, operands[sliceOps]);
      mlir::Value inputStride = *strideOps; // already idxTy
      // Apply origin shift: base += (lb-shift)*input_stride
      mlir::Value sliceOrigin =
          sliceHasOrigins
              ? integerCast(loc, rewriter, idxTy, operands[shiftOps])
              : one;
      mlir::Value diff =
          rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, sliceLb, sliceOrigin);
      mlir::Value offset =
          rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, inputStride);
      base = genGEP(loc, voidPtrTy, rewriter, base, offset);
      // Apply upper bound and step if this is a triplet. Otherwise, the
      // dimension is dropped and no extents/strides are computed.
      mlir::Value upper = operands[sliceOps + 1];
      const bool isTripletSlice =
          !mlir::isa_and_nonnull<mlir::LLVM::UndefOp>(upper.getDefiningOp());
      if (isTripletSlice) {
        mlir::Value step =
            integerCast(loc, rewriter, idxTy, operands[sliceOps + 2]);
        // extent = ub-lb+step/step
        mlir::Value sliceUb = integerCast(loc, rewriter, idxTy, upper);
        mlir::Value extent = computeTripletExtent(rewriter, loc, sliceLb,
                                                  sliceUb, step, zero, idxTy);
        slicedExtents.emplace_back(extent);
        // stride = step*input_stride
        mlir::Value stride =
            rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, step, inputStride);
        slicedStrides.emplace_back(stride);
      }
    }
    return finalizeRebox(rebox, destBoxTy, dest, base, /*lbounds*/ std::nullopt,
                         slicedExtents, slicedStrides, rewriter);
  }

  /// Apply a new shape to the data described by a box given the base address,
  /// extents and strides of the box.
  mlir::LogicalResult
  reshapeBox(fir::cg::XReboxOp rebox, mlir::Type destBoxTy, mlir::Value dest,
             mlir::Value base, mlir::ValueRange inputExtents,
             mlir::ValueRange inputStrides, mlir::ValueRange operands,
             mlir::ConversionPatternRewriter &rewriter) const {
    mlir::ValueRange reboxShifts{operands.begin() + rebox.shiftOffset(),
                                 operands.begin() + rebox.shiftOffset() +
                                     rebox.getShift().size()};
    if (rebox.getShape().empty()) {
      // Only setting new lower bounds.
      return finalizeRebox(rebox, destBoxTy, dest, base, reboxShifts,
                           inputExtents, inputStrides, rewriter);
    }

    mlir::Location loc = rebox.getLoc();
    // Strides from the fir.box are in bytes.
    mlir::Type voidPtrTy = ::getVoidPtrType(rebox.getContext());
    base = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, base);

    llvm::SmallVector<mlir::Value> newStrides;
    llvm::SmallVector<mlir::Value> newExtents;
    mlir::Type idxTy = lowerTy().indexType();
    // First stride from input box is kept. The rest is assumed contiguous
    // (it is not possible to reshape otherwise). If the input is scalar,
    // which may be OK if all new extents are ones, the stride does not
    // matter, use one.
    mlir::Value stride = inputStrides.empty()
                             ? genConstantIndex(loc, idxTy, rewriter, 1)
                             : inputStrides[0];
    for (unsigned i = 0; i < rebox.getShape().size(); ++i) {
      mlir::Value rawExtent = operands[rebox.shapeOffset() + i];
      mlir::Value extent = integerCast(loc, rewriter, idxTy, rawExtent);
      newExtents.emplace_back(extent);
      newStrides.emplace_back(stride);
      // nextStride = extent * stride;
      stride = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, extent, stride);
    }
    return finalizeRebox(rebox, destBoxTy, dest, base, reboxShifts, newExtents,
                         newStrides, rewriter);
  }

  /// Return scalar element type of the input box.
  static mlir::Type getInputEleTy(fir::cg::XReboxOp rebox) {
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(rebox.getBox().getType());
    if (auto seqTy = ty.dyn_cast<fir::SequenceType>())
      return seqTy.getEleTy();
    return ty;
  }
};

/// Lower `fir.emboxproc` operation. Creates a procedure box.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct EmboxProcOpConversion : public FIROpConversion<fir::EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::EmboxProcOp emboxproc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(emboxproc.getLoc(), "fir.emboxproc codegen");
    return mlir::failure();
  }
};

// Code shared between insert_value and extract_value Ops.
struct ValueOpCommon {
  // Translate the arguments pertaining to any multidimensional array to
  // row-major order for LLVM-IR.
  static void toRowMajor(llvm::SmallVectorImpl<int64_t> &indices,
                         mlir::Type ty) {
    assert(ty && "type is null");
    const auto end = indices.size();
    for (std::remove_const_t<decltype(end)> i = 0; i < end; ++i) {
      if (auto seq = ty.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
        const auto dim = getDimension(seq);
        if (dim > 1) {
          auto ub = std::min(i + dim, end);
          std::reverse(indices.begin() + i, indices.begin() + ub);
          i += dim - 1;
        }
        ty = getArrayElementType(seq);
      } else if (auto st = ty.dyn_cast<mlir::LLVM::LLVMStructType>()) {
        ty = st.getBody()[indices[i]];
      } else {
        llvm_unreachable("index into invalid type");
      }
    }
  }

  static llvm::SmallVector<int64_t>
  collectIndices(mlir::ConversionPatternRewriter &rewriter,
                 mlir::ArrayAttr arrAttr) {
    llvm::SmallVector<int64_t> indices;
    for (auto i = arrAttr.begin(), e = arrAttr.end(); i != e; ++i) {
      if (auto intAttr = i->dyn_cast<mlir::IntegerAttr>()) {
        indices.push_back(intAttr.getInt());
      } else {
        auto fieldName = i->cast<mlir::StringAttr>().getValue();
        ++i;
        auto ty = i->cast<mlir::TypeAttr>().getValue();
        auto index = ty.cast<fir::RecordType>().getFieldIndex(fieldName);
        indices.push_back(index);
      }
    }
    return indices;
  }

private:
  static mlir::Type getArrayElementType(mlir::LLVM::LLVMArrayType ty) {
    auto eleTy = ty.getElementType();
    while (auto arrTy = eleTy.dyn_cast<mlir::LLVM::LLVMArrayType>())
      eleTy = arrTy.getElementType();
    return eleTy;
  }
};

namespace {
/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion
    : public FIROpAndTypeConversion<fir::ExtractValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::ExtractValueOp extractVal, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    auto indices = collectIndices(rewriter, extractVal.getCoor());
    toRowMajor(indices, operands[0].getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        extractVal, operands[0], indices);
    return mlir::success();
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
    mlir::ValueRange operands = adaptor.getOperands();
    auto indices = collectIndices(rewriter, insertVal.getCoor());
    toRowMajor(indices, operands[0].getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        insertVal, operands[0], operands[1], indices);
    return mlir::success();
  }
};

/// InsertOnRange inserts a value into a sequence over a range of offsets.
struct InsertOnRangeOpConversion
    : public FIROpAndTypeConversion<fir::InsertOnRangeOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  // Increments an array of subscripts in a row major fasion.
  void incrementSubscripts(llvm::ArrayRef<int64_t> dims,
                           llvm::SmallVectorImpl<int64_t> &subscripts) const {
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

    llvm::SmallVector<std::int64_t> dims;
    auto type = adaptor.getOperands()[0].getType();

    // Iteratively extract the array dimensions from the type.
    while (auto t = type.dyn_cast<mlir::LLVM::LLVMArrayType>()) {
      dims.push_back(t.getNumElements());
      type = t.getElementType();
    }

    llvm::SmallVector<std::int64_t> lBounds;
    llvm::SmallVector<std::int64_t> uBounds;

    // Unzip the upper and lower bound and convert to a row major format.
    mlir::DenseIntElementsAttr coor = range.getCoor();
    auto reversedCoor = llvm::reverse(coor.getValues<int64_t>());
    for (auto i = reversedCoor.begin(), e = reversedCoor.end(); i != e; ++i) {
      uBounds.push_back(*i++);
      lBounds.push_back(*i);
    }

    auto &subscripts = lBounds;
    auto loc = range.getLoc();
    mlir::Value lastOp = adaptor.getOperands()[0];
    mlir::Value insertVal = adaptor.getOperands()[1];

    while (subscripts != uBounds) {
      lastOp = rewriter.create<mlir::LLVM::InsertValueOp>(
          loc, lastOp, insertVal, subscripts);

      incrementSubscripts(dims, subscripts);
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        range, lastOp, insertVal, subscripts);

    return mlir::success();
  }
};
} // namespace

namespace {
/// XArrayCoor is the address arithmetic on a dynamically shaped, sliced,
/// shifted etc. array.
/// (See the static restriction on coordinate_of.) array_coor determines the
/// coordinate (location) of a specific element.
struct XArrayCoorOpConversion
    : public FIROpAndTypeConversion<fir::cg::XArrayCoorOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::cg::XArrayCoorOp coor, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    mlir::ValueRange operands = adaptor.getOperands();
    unsigned rank = coor.getRank();
    assert(coor.getIndices().size() == rank);
    assert(coor.getShape().empty() || coor.getShape().size() == rank);
    assert(coor.getShift().empty() || coor.getShift().size() == rank);
    assert(coor.getSlice().empty() || coor.getSlice().size() == 3 * rank);
    mlir::Type idxTy = lowerTy().indexType();
    unsigned indexOffset = coor.indicesOffset();
    unsigned shapeOffset = coor.shapeOffset();
    unsigned shiftOffset = coor.shiftOffset();
    unsigned sliceOffset = coor.sliceOffset();
    auto sliceOps = coor.getSlice().begin();
    mlir::Value one = genConstantIndex(loc, idxTy, rewriter, 1);
    mlir::Value prevExt = one;
    mlir::Value offset = genConstantIndex(loc, idxTy, rewriter, 0);
    const bool isShifted = !coor.getShift().empty();
    const bool isSliced = !coor.getSlice().empty();
    const bool baseIsBoxed = coor.getMemref().getType().isa<fir::BaseBoxType>();

    // For each dimension of the array, generate the offset calculation.
    for (unsigned i = 0; i < rank; ++i, ++indexOffset, ++shapeOffset,
                  ++shiftOffset, sliceOffset += 3, sliceOps += 3) {
      mlir::Value index =
          integerCast(loc, rewriter, idxTy, operands[indexOffset]);
      mlir::Value lb =
          isShifted ? integerCast(loc, rewriter, idxTy, operands[shiftOffset])
                    : one;
      mlir::Value step = one;
      bool normalSlice = isSliced;
      // Compute zero based index in dimension i of the element, applying
      // potential triplets and lower bounds.
      if (isSliced) {
        mlir::Value originalUb = *(sliceOps + 1);
        normalSlice =
            !mlir::isa_and_nonnull<fir::UndefOp>(originalUb.getDefiningOp());
        if (normalSlice)
          step = integerCast(loc, rewriter, idxTy, operands[sliceOffset + 2]);
      }
      auto idx = rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, index, lb);
      mlir::Value diff =
          rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, idx, step);
      if (normalSlice) {
        mlir::Value sliceLb =
            integerCast(loc, rewriter, idxTy, operands[sliceOffset]);
        auto adj = rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, sliceLb, lb);
        diff = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, diff, adj);
      }
      // Update the offset given the stride and the zero based index `diff`
      // that was just computed.
      if (baseIsBoxed) {
        // Use stride in bytes from the descriptor.
        mlir::Value stride = getStrideFromBox(loc, coor.getMemref().getType(),
                                              operands[0], i, rewriter);
        auto sc = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, stride);
        offset = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, offset);
      } else {
        // Use stride computed at last iteration.
        auto sc = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, prevExt);
        offset = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, offset);
        // Compute next stride assuming contiguity of the base array
        // (in element number).
        auto nextExt = integerCast(loc, rewriter, idxTy, operands[shapeOffset]);
        prevExt =
            rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, prevExt, nextExt);
      }
    }

    // Add computed offset to the base address.
    if (baseIsBoxed) {
      // Working with byte offsets. The base address is read from the fir.box.
      // and need to be casted to i8* to do the pointer arithmetic.
      mlir::Type baseTy = getBaseAddrTypeFromBox(operands[0].getType());
      mlir::Value base = getBaseAddrFromBox(
          loc, baseTy, coor.getMemref().getType(), operands[0], rewriter);
      mlir::Type voidPtrTy = getVoidPtrType();
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, base);
      llvm::SmallVector<mlir::LLVM::GEPArg> args{offset};
      auto addr =
          rewriter.create<mlir::LLVM::GEPOp>(loc, voidPtrTy, base, args);
      if (coor.getSubcomponent().empty()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, ty, addr);
        return mlir::success();
      }
      // Cast the element address from void* to the derived type so that the
      // derived type members can be addresses via a GEP using the index of
      // components.
      mlir::Type elementType =
          baseTy.cast<mlir::LLVM::LLVMPointerType>().getElementType();
      while (auto arrayTy = elementType.dyn_cast<mlir::LLVM::LLVMArrayType>())
        elementType = arrayTy.getElementType();
      mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(elementType);
      auto casted =
          rewriter.create<mlir::LLVM::BitcastOp>(loc, elementPtrType, addr);
      args.clear();
      args.push_back(0);
      if (!coor.getLenParams().empty()) {
        // If type parameters are present, then we don't want to use a GEPOp
        // as below, as the LLVM struct type cannot be statically defined.
        TODO(loc, "derived type with type parameters");
      }
      llvm::SmallVector<mlir::Value> indices = convertSubcomponentIndices(
          loc, elementType,
          operands.slice(coor.subcomponentOffset(),
                         coor.getSubcomponent().size()));
      args.append(indices.begin(), indices.end());
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, ty, casted, args);
      return mlir::success();
    }

    // The array was not boxed, so it must be contiguous. offset is therefore an
    // element offset and the base type is kept in the GEP unless the element
    // type size is itself dynamic.
    mlir::Value base;
    if (coor.getSubcomponent().empty()) {
      // No subcomponent.
      if (!coor.getLenParams().empty()) {
        // Type parameters. Adjust element size explicitly.
        auto eleTy = fir::dyn_cast_ptrEleTy(coor.getType());
        assert(eleTy && "result must be a reference-like type");
        if (fir::characterWithDynamicLen(eleTy)) {
          assert(coor.getLenParams().size() == 1);
          auto length = integerCast(loc, rewriter, idxTy,
                                    operands[coor.lenParamsOffset()]);
          offset =
              rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, offset, length);
        } else {
          TODO(loc, "compute size of derived type with type parameters");
        }
      }
      // Cast the base address to a pointer to T.
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, ty, operands[0]);
    }

    llvm::SmallVector<mlir::LLVM::GEPArg> args = {offset};
    if (!coor.getSubcomponent().empty()) {
      // Operand #0 must have a pointer type. For subcomponent slicing, we
      // want to cast away the array type and have a plain struct type.
      mlir::Type ty0 = operands[0].getType();
      auto ptrTy = ty0.dyn_cast<mlir::LLVM::LLVMPointerType>();
      assert(ptrTy && "expected pointer type");
      mlir::Type eleTy = ptrTy.getElementType();
      while (auto arrTy = eleTy.dyn_cast<mlir::LLVM::LLVMArrayType>())
        eleTy = arrTy.getElementType();
      auto newTy = mlir::LLVM::LLVMPointerType::get(eleTy);
      base = rewriter.create<mlir::LLVM::BitcastOp>(loc, newTy, operands[0]);
      llvm::SmallVector<mlir::Value> indices = convertSubcomponentIndices(
          loc, eleTy,
          operands.slice(coor.subcomponentOffset(),
                         coor.getSubcomponent().size()));
      args.append(indices.begin(), indices.end());
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, ty, base, args);
    return mlir::success();
  }
};
} // namespace

/// Convert to (memory) reference to a reference to a subobject.
/// The coordinate_of op is a Swiss army knife operation that can be used on
/// (memory) references to records, arrays, complex, etc. as well as boxes.
/// With unboxed arrays, there is the restriction that the array have a static
/// shape in all but the last column.
struct CoordinateOpConversion
    : public FIROpAndTypeConversion<fir::CoordinateOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  mlir::LogicalResult
  doRewrite(fir::CoordinateOp coor, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();

    mlir::Location loc = coor.getLoc();
    mlir::Value base = operands[0];
    mlir::Type baseObjectTy = coor.getBaseType();
    mlir::Type objectTy = fir::dyn_cast_ptrOrBoxEleTy(baseObjectTy);
    assert(objectTy && "fir.coordinate_of expects a reference type");

    // Complex type - basically, extract the real or imaginary part
    if (fir::isa_complex(objectTy)) {
      mlir::Value gep = genGEP(loc, ty, rewriter, base, 0, operands[1]);
      rewriter.replaceOp(coor, gep);
      return mlir::success();
    }

    // Boxed type - get the base pointer from the box
    if (baseObjectTy.dyn_cast<fir::BaseBoxType>())
      return doRewriteBox(coor, ty, operands, loc, rewriter);

    // Reference, pointer or a heap type
    if (baseObjectTy.isa<fir::ReferenceType, fir::PointerType, fir::HeapType>())
      return doRewriteRefOrPtr(coor, ty, operands, loc, rewriter);

    return rewriter.notifyMatchFailure(
        coor, "fir.coordinate_of base operand has unsupported type");
  }

  static unsigned getFieldNumber(fir::RecordType ty, mlir::Value op) {
    return fir::hasDynamicSize(ty)
               ? op.getDefiningOp()
                     ->getAttrOfType<mlir::IntegerAttr>("field")
                     .getInt()
               : getConstantIntValue(op);
  }

  static bool hasSubDimensions(mlir::Type type) {
    return type.isa<fir::SequenceType, fir::RecordType, mlir::TupleType>();
  }

  /// Check whether this form of `!fir.coordinate_of` is supported. These
  /// additional checks are required, because we are not yet able to convert
  /// all valid forms of `!fir.coordinate_of`.
  /// TODO: Either implement the unsupported cases or extend the verifier
  /// in FIROps.cpp instead.
  static bool supportedCoordinate(mlir::Type type, mlir::ValueRange coors) {
    const std::size_t numOfCoors = coors.size();
    std::size_t i = 0;
    bool subEle = false;
    bool ptrEle = false;
    for (; i < numOfCoors; ++i) {
      mlir::Value nxtOpnd = coors[i];
      if (auto arrTy = type.dyn_cast<fir::SequenceType>()) {
        subEle = true;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto recTy = type.dyn_cast<fir::RecordType>()) {
        subEle = true;
        type = recTy.getType(getFieldNumber(recTy, nxtOpnd));
      } else if (auto tupTy = type.dyn_cast<mlir::TupleType>()) {
        subEle = true;
        type = tupTy.getType(getConstantIntValue(nxtOpnd));
      } else {
        ptrEle = true;
      }
    }
    if (ptrEle)
      return (!subEle) && (numOfCoors == 1);
    return subEle && (i >= numOfCoors);
  }

  /// Walk the abstract memory layout and determine if the path traverses any
  /// array types with unknown shape. Return true iff all the array types have a
  /// constant shape along the path.
  static bool arraysHaveKnownShape(mlir::Type type, mlir::ValueRange coors) {
    for (std::size_t i = 0, sz = coors.size(); i < sz; ++i) {
      mlir::Value nxtOpnd = coors[i];
      if (auto arrTy = type.dyn_cast<fir::SequenceType>()) {
        if (fir::sequenceWithNonConstantShape(arrTy))
          return false;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto strTy = type.dyn_cast<fir::RecordType>()) {
        type = strTy.getType(getFieldNumber(strTy, nxtOpnd));
      } else if (auto strTy = type.dyn_cast<mlir::TupleType>()) {
        type = strTy.getType(getConstantIntValue(nxtOpnd));
      } else {
        return true;
      }
    }
    return true;
  }

private:
  mlir::LogicalResult
  doRewriteBox(fir::CoordinateOp coor, mlir::Type ty, mlir::ValueRange operands,
               mlir::Location loc,
               mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Type boxObjTy = coor.getBaseType();
    assert(boxObjTy.dyn_cast<fir::BaseBoxType>() && "This is not a `fir.box`");

    mlir::Value boxBaseAddr = operands[0];

    // 1. SPECIAL CASE (uses `fir.len_param_index`):
    //   %box = ... : !fir.box<!fir.type<derived{len1:i32}>>
    //   %lenp = fir.len_param_index len1, !fir.type<derived{len1:i32}>
    //   %addr = coordinate_of %box, %lenp
    if (coor.getNumOperands() == 2) {
      mlir::Operation *coordinateDef =
          (*coor.getCoor().begin()).getDefiningOp();
      if (mlir::isa_and_nonnull<fir::LenParamIndexOp>(coordinateDef))
        TODO(loc,
             "fir.coordinate_of - fir.len_param_index is not supported yet");
    }

    // 2. GENERAL CASE:
    // 2.1. (`fir.array`)
    //   %box = ... : !fix.box<!fir.array<?xU>>
    //   %idx = ... : index
    //   %resultAddr = coordinate_of %box, %idx : !fir.ref<U>
    // 2.2 (`fir.derived`)
    //   %box = ... : !fix.box<!fir.type<derived_type{field_1:i32}>>
    //   %idx = ... : i32
    //   %resultAddr = coordinate_of %box, %idx : !fir.ref<i32>
    // 2.3 (`fir.derived` inside `fir.array`)
    //   %box = ... : !fir.box<!fir.array<10 x !fir.type<derived_1{field_1:f32,
    //   field_2:f32}>>> %idx1 = ... : index %idx2 = ... : i32 %resultAddr =
    //   coordinate_of %box, %idx1, %idx2 : !fir.ref<f32>
    // 2.4. TODO: Either document or disable any other case that the following
    //  implementation might convert.
    mlir::Value resultAddr =
        getBaseAddrFromBox(loc, getBaseAddrTypeFromBox(boxBaseAddr.getType()),
                           boxObjTy, boxBaseAddr, rewriter);
    // Component Type
    auto cpnTy = fir::dyn_cast_ptrOrBoxEleTy(boxObjTy);
    mlir::Type voidPtrTy = ::getVoidPtrType(coor.getContext());

    for (unsigned i = 1, last = operands.size(); i < last; ++i) {
      if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
        if (i != 1)
          TODO(loc, "fir.array nested inside other array and/or derived type");
        // Applies byte strides from the box. Ignore lower bound from box
        // since fir.coordinate_of indexes are zero based. Lowering takes care
        // of lower bound aspects. This both accounts for dynamically sized
        // types and non contiguous arrays.
        auto idxTy = lowerTy().indexType();
        mlir::Value off = genConstantIndex(loc, idxTy, rewriter, 0);
        for (unsigned index = i, lastIndex = i + arrTy.getDimension();
             index < lastIndex; ++index) {
          mlir::Value stride =
              getStrideFromBox(loc, boxObjTy, operands[0], index - i, rewriter);
          auto sc = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy,
                                                       operands[index], stride);
          off = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, off);
        }
        auto voidPtrBase =
            rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, resultAddr);
        resultAddr = rewriter.create<mlir::LLVM::GEPOp>(
            loc, voidPtrTy, voidPtrBase,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{off});
        i += arrTy.getDimension() - 1;
        cpnTy = arrTy.getEleTy();
      } else if (auto recTy = cpnTy.dyn_cast<fir::RecordType>()) {
        auto recRefTy =
            mlir::LLVM::LLVMPointerType::get(lowerTy().convertType(recTy));
        mlir::Value nxtOpnd = operands[i];
        auto memObj =
            rewriter.create<mlir::LLVM::BitcastOp>(loc, recRefTy, resultAddr);
        cpnTy = recTy.getType(getFieldNumber(recTy, nxtOpnd));
        auto llvmCurrentObjTy = lowerTy().convertType(cpnTy);
        auto gep = rewriter.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(llvmCurrentObjTy), memObj,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{0, nxtOpnd});
        resultAddr =
            rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy, gep);
      } else {
        fir::emitFatalError(loc, "unexpected type in coordinate_of");
      }
    }

    rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, ty, resultAddr);
    return mlir::success();
  }

  mlir::LogicalResult
  doRewriteRefOrPtr(fir::CoordinateOp coor, mlir::Type ty,
                    mlir::ValueRange operands, mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Type baseObjectTy = coor.getBaseType();

    // Component Type
    mlir::Type cpnTy = fir::dyn_cast_ptrOrBoxEleTy(baseObjectTy);
    bool hasSubdimension = hasSubDimensions(cpnTy);
    bool columnIsDeferred = !hasSubdimension;

    if (!supportedCoordinate(cpnTy, operands.drop_front(1)))
      TODO(loc, "unsupported combination of coordinate operands");

    const bool hasKnownShape =
        arraysHaveKnownShape(cpnTy, operands.drop_front(1));

    // If only the column is `?`, then we can simply place the column value in
    // the 0-th GEP position.
    if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
      if (!hasKnownShape) {
        const unsigned sz = arrTy.getDimension();
        if (arraysHaveKnownShape(arrTy.getEleTy(),
                                 operands.drop_front(1 + sz))) {
          fir::SequenceType::ShapeRef shape = arrTy.getShape();
          bool allConst = true;
          for (unsigned i = 0; i < sz - 1; ++i) {
            if (shape[i] < 0) {
              allConst = false;
              break;
            }
          }
          if (allConst)
            columnIsDeferred = true;
        }
      }
    }

    if (fir::hasDynamicSize(fir::unwrapSequenceType(cpnTy)))
      return mlir::emitError(
          loc, "fir.coordinate_of with a dynamic element size is unsupported");

    if (hasKnownShape || columnIsDeferred) {
      llvm::SmallVector<mlir::LLVM::GEPArg> offs;
      if (hasKnownShape && hasSubdimension) {
        offs.push_back(0);
      }
      std::optional<int> dims;
      llvm::SmallVector<mlir::Value> arrIdx;
      for (std::size_t i = 1, sz = operands.size(); i < sz; ++i) {
        mlir::Value nxtOpnd = operands[i];

        if (!cpnTy)
          return mlir::emitError(loc, "invalid coordinate/check failed");

        // check if the i-th coordinate relates to an array
        if (dims) {
          arrIdx.push_back(nxtOpnd);
          int dimsLeft = *dims;
          if (dimsLeft > 1) {
            dims = dimsLeft - 1;
            continue;
          }
          cpnTy = cpnTy.cast<fir::SequenceType>().getEleTy();
          // append array range in reverse (FIR arrays are column-major)
          offs.append(arrIdx.rbegin(), arrIdx.rend());
          arrIdx.clear();
          dims.reset();
          continue;
        }
        if (auto arrTy = cpnTy.dyn_cast<fir::SequenceType>()) {
          int d = arrTy.getDimension() - 1;
          if (d > 0) {
            dims = d;
            arrIdx.push_back(nxtOpnd);
            continue;
          }
          cpnTy = cpnTy.cast<fir::SequenceType>().getEleTy();
          offs.push_back(nxtOpnd);
          continue;
        }

        // check if the i-th coordinate relates to a field
        if (auto recTy = cpnTy.dyn_cast<fir::RecordType>())
          cpnTy = recTy.getType(getFieldNumber(recTy, nxtOpnd));
        else if (auto tupTy = cpnTy.dyn_cast<mlir::TupleType>())
          cpnTy = tupTy.getType(getConstantIntValue(nxtOpnd));
        else
          cpnTy = nullptr;

        offs.push_back(nxtOpnd);
      }
      if (dims)
        offs.append(arrIdx.rbegin(), arrIdx.rend());
      mlir::Value base = operands[0];
      mlir::Value retval = genGEP(loc, ty, rewriter, base, offs);
      rewriter.replaceOp(coor, retval);
      return mlir::success();
    }

    return mlir::emitError(
        loc, "fir.coordinate_of base operand has unsupported type");
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
    auto recTy = field.getOnType().cast<fir::RecordType>();
    unsigned index = recTy.getFieldIndex(field.getFieldId());

    if (!fir::hasDynamicSize(recTy)) {
      // Derived type has compile-time constant layout. Return index of the
      // component type in the parent type (to be used in GEP).
      rewriter.replaceOp(field, mlir::ValueRange{genConstantOffset(
                                    field.getLoc(), rewriter, index)});
      return mlir::success();
    }

    // Derived type has compile-time constant layout. Call the compiler
    // generated function to determine the byte offset of the field at runtime.
    // This returns a non-constant.
    mlir::FlatSymbolRefAttr symAttr = mlir::SymbolRefAttr::get(
        field.getContext(), getOffsetMethodName(recTy, field.getFieldId()));
    mlir::NamedAttribute callAttr = rewriter.getNamedAttr("callee", symAttr);
    mlir::NamedAttribute fieldAttr = rewriter.getNamedAttr(
        "field", mlir::IntegerAttr::get(lowerTy().indexType(), index));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        field, lowerTy().offsetType(), adaptor.getOperands(),
        llvm::ArrayRef<mlir::NamedAttribute>{callAttr, fieldAttr});
    return mlir::success();
  }

  // Re-Construct the name of the compiler generated method that calculates the
  // offset
  inline static std::string getOffsetMethodName(fir::RecordType recTy,
                                                llvm::StringRef field) {
    return recTy.getName().str() + "P." + field.str() + ".offset";
  }
};

/// Convert `fir.end`
struct FirEndOpConversion : public FIROpConversion<fir::FirEndOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::FirEndOp firEnd, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(firEnd.getLoc(), "fir.end codegen");
    return mlir::failure();
  }
};

/// Lower `fir.type_desc` to a global addr.
struct TypeDescOpConversion : public FIROpConversion<fir::TypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::TypeDescOp typeDescOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type inTy = typeDescOp.getInType();
    assert(inTy.isa<fir::RecordType>() && "expecting fir.type");
    auto recordType = inTy.dyn_cast<fir::RecordType>();
    auto module = typeDescOp.getOperation()->getParentOfType<mlir::ModuleOp>();
    std::string typeDescName =
        fir::NameUniquer::getTypeDescriptorName(recordType.getName());
    if (auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(typeDescName)) {
      auto ty = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(global.getType()));
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(typeDescOp, ty,
                                                           global.getSymName());
      return mlir::success();
    } else if (auto global = module.lookupSymbol<fir::GlobalOp>(typeDescName)) {
      auto ty = mlir::LLVM::LLVMPointerType::get(
          this->lowerTy().convertType(global.getType()));
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(typeDescOp, ty,
                                                           global.getSymName());
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Lower `fir.has_value` operation to `llvm.return` operation.
struct HasValueOpConversion : public FIROpConversion<fir::HasValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::HasValueOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::success();
  }
};

#ifndef NDEBUG
// Check if attr's type is compatible with ty.
//
// This is done by comparing attr's element type, converted to LLVM type,
// with ty's element type.
//
// Only integer and floating point (including complex) attributes are
// supported. Also, attr is expected to have a TensorType and ty is expected
// to be of LLVMArrayType. If any of the previous conditions is false, then
// the specified attr and ty are not supported by this function and are
// assumed to be compatible.
static inline bool attributeTypeIsCompatible(mlir::MLIRContext *ctx,
                                             mlir::Attribute attr,
                                             mlir::Type ty) {
  // Get attr's LLVM element type.
  if (!attr)
    return true;
  auto intOrFpEleAttr = mlir::dyn_cast<mlir::DenseIntOrFPElementsAttr>(attr);
  if (!intOrFpEleAttr)
    return true;
  auto tensorTy = mlir::dyn_cast<mlir::TensorType>(intOrFpEleAttr.getType());
  if (!tensorTy)
    return true;
  mlir::Type attrEleTy =
      mlir::LLVMTypeConverter(ctx).convertType(tensorTy.getElementType());

  // Get ty's element type.
  auto arrTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(ty);
  if (!arrTy)
    return true;
  mlir::Type eleTy = arrTy.getElementType();
  while ((arrTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(eleTy)))
    eleTy = arrTy.getElementType();

  return attrEleTy == eleTy;
}
#endif

/// Lower `fir.global` operation to `llvm.global` operation.
/// `fir.insert_on_range` operations are replaced with constant dense attribute
/// if they are applied on the full range.
struct GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::GlobalOp global, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tyAttr = convertType(global.getType());
    if (global.getType().isa<fir::BaseBoxType>())
      tyAttr = tyAttr.cast<mlir::LLVM::LLVMPointerType>().getElementType();
    auto loc = global.getLoc();
    mlir::Attribute initAttr = global.getInitVal().value_or(mlir::Attribute());
    assert(attributeTypeIsCompatible(global.getContext(), initAttr, tyAttr));
    auto linkage = convertLinkage(global.getLinkName());
    auto isConst = global.getConstant().has_value();
    auto g = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, tyAttr, isConst, linkage, global.getSymName(), initAttr);

    auto module = global->getParentOfType<mlir::ModuleOp>();
    // Add comdat if necessary
    if (fir::getTargetTriple(module).supportsCOMDAT() &&
        (linkage == mlir::LLVM::Linkage::Linkonce ||
         linkage == mlir::LLVM::Linkage::LinkonceODR)) {
      addComdat(g, rewriter, module);
    }

    // Apply all non-Fir::GlobalOp attributes to the LLVM::GlobalOp, preserving
    // them; whilst taking care not to apply attributes that are lowered in
    // other ways.
    llvm::SmallDenseSet<llvm::StringRef> elidedAttrsSet(
        global.getAttributeNames().begin(), global.getAttributeNames().end());
    for (auto &attr : global->getAttrs())
      if (!elidedAttrsSet.contains(attr.getName().strref()))
        g->setAttr(attr.getName(), attr.getValue());

    auto &gr = g.getInitializerRegion();
    rewriter.inlineRegionBefore(global.getRegion(), gr, gr.end());
    if (!gr.empty()) {
      // Replace insert_on_range with a constant dense attribute if the
      // initialization is on the full range.
      auto insertOnRangeOps = gr.front().getOps<fir::InsertOnRangeOp>();
      for (auto insertOp : insertOnRangeOps) {
        if (isFullRange(insertOp.getCoor(), insertOp.getType())) {
          auto seqTyAttr = convertType(insertOp.getType());
          auto *op = insertOp.getVal().getDefiningOp();
          auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op);
          if (!constant) {
            auto convertOp = mlir::dyn_cast<fir::ConvertOp>(op);
            if (!convertOp)
              continue;
            constant = mlir::cast<mlir::arith::ConstantOp>(
                convertOp.getValue().getDefiningOp());
          }
          mlir::Type vecType = mlir::VectorType::get(
              insertOp.getType().getShape(), constant.getType());
          auto denseAttr = mlir::DenseElementsAttr::get(
              vecType.cast<mlir::ShapedType>(), constant.getValue());
          rewriter.setInsertionPointAfter(insertOp);
          rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
              insertOp, seqTyAttr, denseAttr);
        }
      }
    }
    rewriter.eraseOp(global);
    return mlir::success();
  }

  bool isFullRange(mlir::DenseIntElementsAttr indexes,
                   fir::SequenceType seqTy) const {
    auto extents = seqTy.getShape();
    if (indexes.size() / 2 != static_cast<int64_t>(extents.size()))
      return false;
    auto cur_index = indexes.value_begin<int64_t>();
    for (unsigned i = 0; i < indexes.size(); i += 2) {
      if (*(cur_index++) != 0)
        return false;
      if (*(cur_index++) != extents[i / 2] - 1)
        return false;
    }
    return true;
  }

  // TODO: String comparaison should be avoided. Replace linkName with an
  // enumeration.
  mlir::LLVM::Linkage
  convertLinkage(std::optional<llvm::StringRef> optLinkage) const {
    if (optLinkage) {
      auto name = *optLinkage;
      if (name == "internal")
        return mlir::LLVM::Linkage::Internal;
      if (name == "linkonce")
        return mlir::LLVM::Linkage::Linkonce;
      if (name == "linkonce_odr")
        return mlir::LLVM::Linkage::LinkonceODR;
      if (name == "common")
        return mlir::LLVM::Linkage::Common;
      if (name == "weak")
        return mlir::LLVM::Linkage::Weak;
    }
    return mlir::LLVM::Linkage::External;
  }

private:
  static void addComdat(mlir::LLVM::GlobalOp &global,
                        mlir::ConversionPatternRewriter &rewriter,
                        mlir::ModuleOp &module) {
    const char *comdatName = "__llvm_comdat";
    mlir::LLVM::ComdatOp comdatOp =
        module.lookupSymbol<mlir::LLVM::ComdatOp>(comdatName);
    if (!comdatOp) {
      comdatOp =
          rewriter.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
    }
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&comdatOp.getBody().back());
    auto selectorOp = rewriter.create<mlir::LLVM::ComdatSelectorOp>(
        comdatOp.getLoc(), global.getSymName(),
        mlir::LLVM::comdat::Comdat::Any);
    global.setComdatAttr(mlir::SymbolRefAttr::get(
        rewriter.getContext(), comdatName,
        mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
  }
};

/// `fir.load` --> `llvm.load`
struct LoadOpConversion : public FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::LoadOp load, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (auto boxTy = load.getType().dyn_cast<fir::BaseBoxType>()) {
      // fir.box is a special case because it is considered as an ssa values in
      // fir, but it is lowered as a pointer to a descriptor. So
      // fir.ref<fir.box> and fir.box end up being the same llvm types and
      // loading a fir.ref<fir.box> is implemented as taking a snapshot of the
      // descriptor value into a new descriptor temp.
      auto inputBoxStorage = adaptor.getOperands()[0];
      mlir::Location loc = load.getLoc();
      fir::SequenceType seqTy = fir::unwrapUntilSeqType(boxTy);
      // fir.box of assumed rank do not have a storage
      // size that is know at compile time. The copy needs to be runtime driven
      // depending on the actual dynamic rank or type.
      if (seqTy && seqTy.hasUnknownShape())
        TODO(loc, "loading or assumed rank fir.box");
      mlir::Type boxPtrTy = inputBoxStorage.getType();
      auto boxValue = rewriter.create<mlir::LLVM::LoadOp>(
          loc, boxPtrTy.cast<mlir::LLVM::LLVMPointerType>().getElementType(),
          inputBoxStorage);
      attachTBAATag(boxValue, boxTy, boxTy, nullptr);
      auto newBoxStorage =
          genAllocaWithType(loc, boxPtrTy, defaultAlign, rewriter);
      auto storeOp =
          rewriter.create<mlir::LLVM::StoreOp>(loc, boxValue, newBoxStorage);
      attachTBAATag(storeOp, boxTy, boxTy, nullptr);
      rewriter.replaceOp(load, newBoxStorage.getResult());
    } else {
      mlir::Type loadTy = convertType(load.getType());
      auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(
          load.getLoc(), loadTy, adaptor.getOperands(), load->getAttrs());
      attachTBAATag(loadOp, load.getType(), load.getType(), nullptr);
      rewriter.replaceOp(load, loadOp.getResult());
    }
    return mlir::success();
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
    return mlir::success();
  }
};

static void genCondBrOp(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                        std::optional<mlir::ValueRange> destOps,
                        mlir::ConversionPatternRewriter &rewriter,
                        mlir::Block *newBlock) {
  if (destOps)
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, *destOps, newBlock,
                                          mlir::ValueRange());
  else
    rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp, dest, newBlock);
}

template <typename A, typename B>
static void genBrOp(A caseOp, mlir::Block *dest, std::optional<B> destOps,
                    mlir::ConversionPatternRewriter &rewriter) {
  if (destOps)
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, *destOps, dest);
  else
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(caseOp, std::nullopt, dest);
}

static void genCaseLadderStep(mlir::Location loc, mlir::Value cmp,
                              mlir::Block *dest,
                              std::optional<mlir::ValueRange> destOps,
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
/// A point value case such as `case(4)`, a lower bound case such as
/// `case(5:)` or an upper bound case such as `case(:3)` are converted to a
/// simple comparison between the selector value and the constant value in the
/// case. The block associated with the case condition is then executed if
/// the comparison succeed otherwise it branch to the next block with the
/// comparison for the next case conditon.
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
    auto ty = caseOp.getSelector().getType();
    if (ty.isa<fir::CharacterType>()) {
      TODO(caseOp.getLoc(), "fir.select_case codegen with character type");
      return mlir::failure();
    }
    mlir::Value selector = caseOp.getSelector(adaptor.getOperands());
    auto loc = caseOp.getLoc();
    for (unsigned t = 0; t != conds; ++t) {
      mlir::Block *dest = caseOp.getSuccessor(t);
      std::optional<mlir::ValueRange> destOps =
          caseOp.getSuccessorOperands(adaptor.getOperands(), t);
      std::optional<mlir::ValueRange> cmpOps =
          *caseOp.getCompareOperands(adaptor.getOperands(), t);
      mlir::Value caseArg = *(cmpOps.value().begin());
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
        mlir::Value caseArg0 = *(cmpOps.value().begin() + 1);
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
    return mlir::success();
  }
};

template <typename OP>
static void selectMatchAndRewrite(const fir::LLVMTypeConverter &lowering,
                                  OP select, typename OP::Adaptor adaptor,
                                  mlir::ConversionPatternRewriter &rewriter) {
  unsigned conds = select.getNumConditions();
  auto cases = select.getCases().getValue();
  mlir::Value selector = adaptor.getSelector();
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
      destinationsOperands.push_back(destOps ? *destOps : mlir::ValueRange{});
      caseValues.push_back(intAttr.getInt());
      continue;
    }
    assert(attr.template dyn_cast_or_null<mlir::UnitAttr>());
    assert((t + 1 == conds) && "unit must be last");
    defaultDestination = dest;
    defaultOperands = destOps ? *destOps : mlir::ValueRange{};
  }

  // LLVM::SwitchOp takes a i32 type for the selector.
  if (select.getSelector().getType() != rewriter.getI32Type())
    selector = rewriter.create<mlir::LLVM::TruncOp>(loc, rewriter.getI32Type(),
                                                    selector);

  rewriter.replaceOpWithNewOp<mlir::LLVM::SwitchOp>(
      select, selector,
      /*defaultDestination=*/defaultDestination,
      /*defaultOperands=*/defaultOperands,
      /*caseValues=*/caseValues,
      /*caseDestinations=*/destinations,
      /*caseOperands=*/destinationsOperands,
      /*branchWeights=*/llvm::ArrayRef<std::int32_t>());
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectOp>(lowerTy(), op, adaptor, rewriter);
    return mlir::success();
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public FIROpConversion<fir::SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectRankOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectRankOp>(lowerTy(), op, adaptor, rewriter);
    return mlir::success();
  }
};

/// Lower `fir.select_type` to LLVM IR dialect.
struct SelectTypeOpConversion : public FIROpConversion<fir::SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::SelectTypeOp select, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::emitError(select.getLoc(),
                    "fir.select_type should have already been converted");
    return mlir::failure();
  }
};

/// `fir.store` --> `llvm.store`
struct StoreOpConversion : public FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::StoreOp store, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = store.getLoc();
    mlir::Type storeTy = store.getValue().getType();
    mlir::LLVM::StoreOp newStoreOp;
    if (auto boxTy = storeTy.dyn_cast<fir::BaseBoxType>()) {
      // fir.box value is actually in memory, load it first before storing it.
      mlir::Type boxPtrTy = adaptor.getOperands()[0].getType();
      auto val = rewriter.create<mlir::LLVM::LoadOp>(
          loc, boxPtrTy.cast<mlir::LLVM::LLVMPointerType>().getElementType(),
          adaptor.getOperands()[0]);
      attachTBAATag(val, boxTy, boxTy, nullptr);
      newStoreOp = rewriter.create<mlir::LLVM::StoreOp>(
          loc, val, adaptor.getOperands()[1]);
    } else {
      newStoreOp = rewriter.create<mlir::LLVM::StoreOp>(
          loc, adaptor.getOperands()[0], adaptor.getOperands()[1]);
    }
    attachTBAATag(newStoreOp, storeTy, storeTy, nullptr);
    rewriter.eraseOp(store);
    return mlir::success();
  }
};

namespace {

/// Convert `fir.unboxchar` into two `llvm.extractvalue` instructions. One for
/// the character buffer and one for the buffer length.
struct UnboxCharOpConversion : public FIROpConversion<fir::UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnboxCharOp unboxchar, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type lenTy = convertType(unboxchar.getType(1));
    mlir::Value tuple = adaptor.getOperands()[0];

    mlir::Location loc = unboxchar.getLoc();
    mlir::Value ptrToBuffer =
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, tuple, 0);

    auto len = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, tuple, 1);
    mlir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, len);

    rewriter.replaceOp(unboxchar,
                       llvm::ArrayRef<mlir::Value>{ptrToBuffer, lenAfterCast});
    return mlir::success();
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
    TODO(unboxproc.getLoc(), "fir.unboxproc codegen");
    return mlir::failure();
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
    return mlir::success();
  }
};

struct ZeroOpConversion : public FIROpConversion<fir::ZeroOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::ZeroOp zero, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = convertType(zero.getType());
    if (ty.isa<mlir::LLVM::LLVMPointerType>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(zero, ty);
    } else if (ty.isa<mlir::IntegerType>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
          zero, ty, mlir::IntegerAttr::get(ty, 0));
    } else if (mlir::LLVM::isCompatibleFloatingPointType(ty)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
          zero, ty, mlir::FloatAttr::get(zero.getType(), 0.0));
    } else {
      // TODO: create ConstantAggregateZero for FIR aggregate/array types.
      return rewriter.notifyMatchFailure(
          zero,
          "conversion of fir.zero with aggregate type not implemented yet");
    }
    return mlir::success();
  }
};

/// `fir.unreachable` --> `llvm.unreachable`
struct UnreachableOpConversion : public FIROpConversion<fir::UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(fir::UnreachableOp unreach, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(unreach);
    return mlir::success();
  }
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

    if (isPresent.getVal().getType().isa<fir::BoxCharType>()) {
      [[maybe_unused]] auto structTy =
          ptr.getType().cast<mlir::LLVM::LLVMStructType>();
      assert(!structTy.isOpaque() && !structTy.getBody().empty());

      ptr = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ptr, 0);
    }
    mlir::LLVM::ConstantOp c0 =
        genConstantIndex(isPresent.getLoc(), idxTy, rewriter, 0);
    auto addr = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, idxTy, ptr);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        isPresent, mlir::LLVM::ICmpPredicate::ne, addr, c0);

    return mlir::success();
  }
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
      rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
          absent, undefStruct, nullField, 0);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::NullOp>(absent, ty);
    }
    return mlir::success();
  }
};

//
// Primitive operations on Complex types
//

/// Generate inline code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
static mlir::LLVM::InsertValueOp
complexSum(OPTY sumop, mlir::ValueRange opnds,
           mlir::ConversionPatternRewriter &rewriter,
           const fir::LLVMTypeConverter &lowering) {
  mlir::Value a = opnds[0];
  mlir::Value b = opnds[1];
  auto loc = sumop.getLoc();
  mlir::Type eleTy = lowering.convertType(getComplexEleTy(sumop.getType()));
  mlir::Type ty = lowering.convertType(sumop.getType());
  auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 0);
  auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 1);
  auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 0);
  auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 1);
  auto rx = rewriter.create<LLVMOP>(loc, eleTy, x0, x1);
  auto ry = rewriter.create<LLVMOP>(loc, eleTy, y0, y1);
  auto r0 = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
  auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, r0, rx, 0);
  return rewriter.create<mlir::LLVM::InsertValueOp>(loc, r1, ry, 1);
}
} // namespace

namespace {
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
    return mlir::success();
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
    return mlir::success();
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
    mlir::Type eleTy = convertType(getComplexEleTy(mulc.getType()));
    mlir::Type ty = convertType(mulc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1);
    auto ri = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xy, yx);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1);
    auto rr = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, xx, yy);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ra, rr, 0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, r1, ri, 1);
    rewriter.replaceOp(mulc, r0.getResult());
    return mlir::success();
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
    mlir::Type eleTy = convertType(getComplexEleTy(divc.getType()));
    mlir::Type ty = convertType(divc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 1);
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
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ra, rr, 0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, r1, ri, 1);
    rewriter.replaceOp(divc, r0.getResult());
    return mlir::success();
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
    auto eleTy = convertType(getComplexEleTy(neg.getType()));
    auto loc = neg.getLoc();
    mlir::Value o0 = adaptor.getOperands()[0];
    auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, o0, 0);
    auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, o0, 1);
    auto nrp = rewriter.create<mlir::LLVM::FNegOp>(loc, eleTy, rp);
    auto nip = rewriter.create<mlir::LLVM::FNegOp>(loc, eleTy, ip);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, o0, nrp, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(neg, r, nip, 1);
    return mlir::success();
  }
};

/// Conversion pattern for operation that must be dead. The information in these
/// operations is used by other operation. At this point they should not have
/// anymore uses.
/// These operations are normally dead after the pre-codegen pass.
template <typename FromOp>
struct MustBeDeadConversion : public FIROpConversion<FromOp> {
  explicit MustBeDeadConversion(const fir::LLVMTypeConverter &lowering,
                                const fir::FIRToLLVMPassOptions &options)
      : FIROpConversion<FromOp>(lowering, options) {}
  using OpAdaptor = typename FromOp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(FromOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (!op->getUses().empty())
      return rewriter.notifyMatchFailure(op, "op must be dead");
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct UnrealizedConversionCastOpConversion
    : public FIROpConversion<mlir::UnrealizedConversionCastOp> {
  using FIROpConversion::FIROpConversion;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(op.getOutputs().getTypes().size() == 1 && "expect a single type");
    mlir::Type convertedType = convertType(op.getOutputs().getTypes()[0]);
    if (convertedType == adaptor.getInputs().getTypes()[0]) {
      rewriter.replaceOp(op, adaptor.getInputs());
      return mlir::success();
    }

    convertedType = adaptor.getInputs().getTypes()[0];
    if (convertedType == op.getOutputs().getType()[0]) {
      rewriter.replaceOp(op, adaptor.getInputs());
      return mlir::success();
    }
    return mlir::failure();
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

} // namespace

namespace {
class RenameMSVCLibmCallees
    : public mlir::OpRewritePattern<mlir::LLVM::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto callee = op.getCallee();
    if (callee)
      if (callee->equals("hypotf"))
        op.setCalleeAttr(mlir::SymbolRefAttr::get(op.getContext(), "_hypotf"));

    rewriter.finalizeRootUpdate(op);
    return mlir::success();
  }
};

class RenameMSVCLibmFuncs
    : public mlir::OpRewritePattern<mlir::LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::LLVM::LLVMFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    if (op.getSymName().equals("hypotf"))
      op.setSymNameAttr(rewriter.getStringAttr("_hypotf"));
    rewriter.finalizeRootUpdate(op);
    return mlir::success();
  }
};
} // namespace

namespace {
/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect. An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
class FIRToLLVMLowering
    : public fir::impl::FIRToLLVMLoweringBase<FIRToLLVMLowering> {
public:
  FIRToLLVMLowering() = default;
  FIRToLLVMLowering(fir::FIRToLLVMPassOptions options) : options{options} {}
  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto mod = getModule();
    if (!forcedTargetTriple.empty())
      fir::setTargetTriple(mod, forcedTargetTriple);

    // Run dynamic pass pipeline for converting Math dialect
    // operations into other dialects (llvm, func, etc.).
    // Some conversions of Math operations cannot be done
    // by just using conversion patterns. This is true for
    // conversions that affect the ModuleOp, e.g. create new
    // function operations in it. We have to run such conversions
    // as passes here.
    mlir::OpPassManager mathConvertionPM("builtin.module");

    // Convert math::FPowI operations to inline implementation
    // only if the exponent's width is greater than 32, otherwise,
    // it will be lowered to LLVM intrinsic operation by a later conversion.
    mlir::ConvertMathToFuncsOptions mathToFuncsOptions{};
    mathToFuncsOptions.minWidthOfFPowIExponent = 33;
    mathConvertionPM.addPass(
        mlir::createConvertMathToFuncs(mathToFuncsOptions));
    mathConvertionPM.addPass(mlir::createConvertComplexToStandardPass());
    // Convert Math dialect operations into LLVM dialect operations.
    // There is no way to prefer MathToLLVM patterns over MathToLibm
    // patterns (applied below), so we have to run MathToLLVM conversion here.
    mathConvertionPM.addNestedPass<mlir::func::FuncOp>(
        mlir::createConvertMathToLLVMPass());
    if (mlir::failed(runPipeline(mathConvertionPM, mod)))
      return signalPassFailure();

    auto *context = getModule().getContext();
    fir::LLVMTypeConverter typeConverter{getModule(),
                                         options.applyTBAA || applyTBAA};
    mlir::RewritePatternSet pattern(context);
    pattern.insert<
        AbsentOpConversion, AddcOpConversion, AddrOfOpConversion,
        AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
        BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
        BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
        BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeCodeOpConversion,
        BoxTypeDescOpConversion, CallOpConversion, CmpcOpConversion,
        ConstcOpConversion, ConvertOpConversion, CoordinateOpConversion,
        DispatchTableOpConversion, DTEntryOpConversion, DivcOpConversion,
        EmboxOpConversion, EmboxCharOpConversion, EmboxProcOpConversion,
        ExtractValueOpConversion, FieldIndexOpConversion, FirEndOpConversion,
        FreeMemOpConversion, GlobalLenOpConversion, GlobalOpConversion,
        HasValueOpConversion, InsertOnRangeOpConversion,
        InsertValueOpConversion, IsPresentOpConversion,
        LenParamIndexOpConversion, LoadOpConversion, MulcOpConversion,
        NegcOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
        SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
        ShapeOpConversion, ShapeShiftOpConversion, ShiftOpConversion,
        SliceOpConversion, StoreOpConversion, StringLitOpConversion,
        SubcOpConversion, TypeDescOpConversion, UnboxCharOpConversion,
        UnboxProcOpConversion, UndefOpConversion, UnreachableOpConversion,
        UnrealizedConversionCastOpConversion, XArrayCoorOpConversion,
        XEmboxOpConversion, XReboxOpConversion, ZeroOpConversion>(typeConverter,
                                                                  options);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, pattern);
    mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, pattern);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, pattern);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          pattern);
    // Math operations that have not been converted yet must be converted
    // to Libm.
    mlir::populateMathToLibmConversionPatterns(pattern);
    mlir::populateComplexToLLVMConversionPatterns(typeConverter, pattern);
    mlir::populateVectorToLLVMConversionPatterns(typeConverter, pattern);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    // The OpenMP dialect is legal for Operations without regions, for those
    // which contains regions it is legal if the region contains only the
    // LLVM dialect. Add OpenMP dialect as a legal dialect for conversion and
    // legalize conversion of OpenMP operations without regions.
    mlir::configureOpenMPToLLVMConversionLegality(target, typeConverter);
    target.addLegalDialect<mlir::omp::OpenMPDialect>();
    target.addLegalDialect<mlir::acc::OpenACCDialect>();

    // required NOPs for applying a full conversion
    target.addLegalOp<mlir::ModuleOp>();

    // If we're on Windows, we might need to rename some libm calls.
    bool isMSVC = fir::getTargetTriple(mod).isOSMSVCRT();
    if (isMSVC) {
      pattern.insert<RenameMSVCLibmCallees, RenameMSVCLibmFuncs>(context);

      target.addDynamicallyLegalOp<mlir::LLVM::CallOp>(
          [](mlir::LLVM::CallOp op) {
            auto callee = op.getCallee();
            if (!callee)
              return true;
            return !callee->equals("hypotf");
          });
      target.addDynamicallyLegalOp<mlir::LLVM::LLVMFuncOp>(
          [](mlir::LLVM::LLVMFuncOp op) {
            return !op.getSymName().equals("hypotf");
          });
    }

    // apply the patterns
    if (mlir::failed(mlir::applyFullConversion(getModule(), target,
                                               std::move(pattern)))) {
      signalPassFailure();
    }
  }

private:
  fir::FIRToLLVMPassOptions options;
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass
    : public mlir::PassWrapper<LLVMIRLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMIRLoweringPass)

  LLVMIRLoweringPass(llvm::raw_ostream &output, fir::LLVMIRLoweringPrinter p)
      : output{output}, printer{p} {}

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto *ctx = getModule().getContext();
    auto optName = getModule().getName();
    llvm::LLVMContext llvmCtx;
    if (auto llvmModule = mlir::translateModuleToLLVMIR(
            getModule(), llvmCtx, optName ? *optName : "FIRModule")) {
      printer(*llvmModule, output);
      return;
    }

    mlir::emitError(mlir::UnknownLoc::get(ctx), "could not emit LLVM-IR\n");
    signalPassFailure();
  }

private:
  llvm::raw_ostream &output;
  fir::LLVMIRLoweringPrinter printer;
};

} // namespace

std::unique_ptr<mlir::Pass> fir::createFIRToLLVMPass() {
  return std::make_unique<FIRToLLVMLowering>();
}

std::unique_ptr<mlir::Pass>
fir::createFIRToLLVMPass(fir::FIRToLLVMPassOptions options) {
  return std::make_unique<FIRToLLVMLowering>(options);
}

std::unique_ptr<mlir::Pass>
fir::createLLVMDialectToLLVMPass(llvm::raw_ostream &output,
                                 fir::LLVMIRLoweringPrinter printer) {
  return std::make_unique<LLVMIRLoweringPass>(output, printer);
}
