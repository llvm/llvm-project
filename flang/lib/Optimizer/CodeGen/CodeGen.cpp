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

#include "flang/Optimizer/CodeGen/CGOps.h"
#include "flang/Optimizer/CodeGen/CodeGenOpenMP.h"
#include "flang/Optimizer/CodeGen/FIROpPatterns.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/TypeCode.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memory.h"
#include "flang/Runtime/allocator-registry-consts.h"
#include "flang/Runtime/descriptor-consts.h"
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
#include "mlir/Conversion/MathToROCDL/MathToROCDL.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/AddComdats.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

namespace fir {
#define GEN_PASS_DEF_FIRTOLLVMLOWERING
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-codegen"

// TODO: This should really be recovered from the specified target.
static constexpr unsigned defaultAlign = 8;

/// `fir.box` attribute values as defined for CFI_attribute_t in
/// flang/ISO_Fortran_binding.h.
static constexpr unsigned kAttrPointer = CFI_attribute_pointer;
static constexpr unsigned kAttrAllocatable = CFI_attribute_allocatable;

static inline mlir::Type getLlvmPtrType(mlir::MLIRContext *context,
                                        unsigned addressSpace = 0) {
  return mlir::LLVM::LLVMPointerType::get(context, addressSpace);
}

static inline mlir::Type getI8Type(mlir::MLIRContext *context) {
  return mlir::IntegerType::get(context, 8);
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

/// Extract constant from a value that must be the result of one of the
/// ConstantOp operations.
static int64_t getConstantIntValue(mlir::Value val) {
  if (auto constVal = fir::getIntIfConstant(val))
    return *constVal;
  fir::emitFatalError(val.getLoc(), "must be a constant");
}

static unsigned getTypeDescFieldId(mlir::Type ty) {
  auto isArray = mlir::isa<fir::SequenceType>(fir::dyn_cast_ptrOrBoxEleTy(ty));
  return isArray ? kOptTypePtrPosInBox : kDimsPosInBox;
}
static unsigned getLenParamFieldId(mlir::Type ty) {
  return getTypeDescFieldId(ty) + 1;
}

static llvm::SmallVector<mlir::NamedAttribute>
addLLVMOpBundleAttrs(mlir::ConversionPatternRewriter &rewriter,
                     llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     int32_t numCallOperands) {
  llvm::SmallVector<mlir::NamedAttribute> newAttrs;
  newAttrs.reserve(attrs.size() + 2);

  for (mlir::NamedAttribute attr : attrs) {
    if (attr.getName() != "operandSegmentSizes")
      newAttrs.push_back(attr);
  }

  newAttrs.push_back(rewriter.getNamedAttr(
      "operandSegmentSizes",
      rewriter.getDenseI32ArrayAttr({numCallOperands, 0})));
  newAttrs.push_back(rewriter.getNamedAttr("op_bundle_sizes",
                                           rewriter.getDenseI32ArrayAttr({})));
  return newAttrs;
}

namespace {
/// Lower `fir.address_of` operation to `llvm.address_of` operation.
struct AddrOfOpConversion : public fir::FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(dataTy);
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
struct DeclareOpConversion : public fir::FIROpConversion<fir::cg::XDeclareOp> {
public:
  using FIROpConversion::FIROpConversion;
  llvm::LogicalResult
  matchAndRewrite(fir::cg::XDeclareOp declareOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memRef = adaptor.getOperands()[0];
    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(declareOp.getLoc())) {
      if (auto varAttr =
              mlir::dyn_cast_or_null<mlir::LLVM::DILocalVariableAttr>(
                  fusedLoc.getMetadata())) {
        rewriter.create<mlir::LLVM::DbgDeclareOp>(memRef.getLoc(), memRef,
                                                  varAttr, nullptr);
      }
    }
    rewriter.replaceOp(declareOp, memRef);
    return mlir::success();
  }
};
} // namespace

namespace {
/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public fir::FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AllocaOp alloc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    auto loc = alloc.getLoc();
    mlir::Type ity = lowerTy().indexType();
    unsigned i = 0;
    mlir::Value size = genConstantIndex(loc, ity, rewriter, 1).getResult();
    mlir::Type firObjType = fir::unwrapRefType(alloc.getType());
    mlir::Type llvmObjectType = convertObjectType(firObjType);
    if (alloc.hasLenParams()) {
      unsigned end = alloc.numLenParams();
      llvm::SmallVector<mlir::Value> lenParams;
      for (; i < end; ++i)
        lenParams.push_back(operands[i]);
      mlir::Type scalarType = fir::unwrapSequenceType(alloc.getInType());
      if (auto chrTy = mlir::dyn_cast<fir::CharacterType>(scalarType)) {
        fir::CharacterType rawCharTy = fir::CharacterType::getUnknownLen(
            chrTy.getContext(), chrTy.getFKind());
        llvmObjectType = convertType(rawCharTy);
        assert(end == 1);
        size = integerCast(loc, rewriter, ity, lenParams[0], /*fold=*/true);
      } else if (auto recTy = mlir::dyn_cast<fir::RecordType>(scalarType)) {
        mlir::LLVM::LLVMFuncOp memSizeFn =
            getDependentTypeMemSizeFn(recTy, alloc, rewriter);
        if (!memSizeFn)
          emitError(loc, "did not find allocation function");
        mlir::NamedAttribute attr = rewriter.getNamedAttr(
            "callee", mlir::SymbolRefAttr::get(memSizeFn));
        auto call = rewriter.create<mlir::LLVM::CallOp>(
            loc, ity, lenParams,
            addLLVMOpBundleAttrs(rewriter, {attr}, lenParams.size()));
        size = call.getResult();
        llvmObjectType = ::getI8Type(alloc.getContext());
      } else {
        return emitError(loc, "unexpected type ")
               << scalarType << " with type parameters";
      }
    }
    if (auto scaleSize = genAllocationScaleSize(alloc, ity, rewriter))
      size =
          rewriter.createOrFold<mlir::LLVM::MulOp>(loc, ity, size, scaleSize);
    if (alloc.hasShapeOperands()) {
      unsigned end = operands.size();
      for (; i < end; ++i)
        size = rewriter.createOrFold<mlir::LLVM::MulOp>(
            loc, ity, size,
            integerCast(loc, rewriter, ity, operands[i], /*fold=*/true));
    }

    unsigned allocaAs = getAllocaAddressSpace(rewriter);
    unsigned programAs = getProgramAddressSpace(rewriter);

    if (mlir::isa<mlir::LLVM::ConstantOp>(size.getDefiningOp())) {
      // Set the Block in which the llvm alloca should be inserted.
      mlir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
      mlir::Region *parentRegion = rewriter.getInsertionBlock()->getParent();
      mlir::Block *insertBlock =
          getBlockForAllocaInsert(parentOp, parentRegion);

      // The old size might have had multiple users, some at a broader scope
      // than we can safely outline the alloca to. As it is only an
      // llvm.constant operation, it is faster to clone it than to calculate the
      // dominance to see if it really should be moved.
      mlir::Operation *clonedSize = rewriter.clone(*size.getDefiningOp());
      size = clonedSize->getResult(0);
      clonedSize->moveBefore(&insertBlock->front());
      rewriter.setInsertionPointAfter(size.getDefiningOp());
    }

    // NOTE: we used to pass alloc->getAttrs() in the builder for non opaque
    // pointers! Only propagate pinned and bindc_name to help debugging, but
    // this should have no functional purpose (and passing the operand segment
    // attribute like before is certainly bad).
    auto llvmAlloc = rewriter.create<mlir::LLVM::AllocaOp>(
        loc, ::getLlvmPtrType(alloc.getContext(), allocaAs), llvmObjectType,
        size);
    if (alloc.getPinned())
      llvmAlloc->setDiscardableAttr(alloc.getPinnedAttrName(),
                                    alloc.getPinnedAttr());
    if (alloc.getBindcName())
      llvmAlloc->setDiscardableAttr(alloc.getBindcNameAttrName(),
                                    alloc.getBindcNameAttr());
    if (allocaAs == programAs) {
      rewriter.replaceOp(alloc, llvmAlloc);
    } else {
      // if our allocation address space, is not the same as the program address
      // space, then we must emit a cast to the program address space before
      // use. An example case would be on AMDGPU, where the allocation address
      // space is the numeric value 5 (private), and the program address space
      // is 0 (generic).
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddrSpaceCastOp>(
          alloc, ::getLlvmPtrType(alloc.getContext(), programAs), llvmAlloc);
    }
    return mlir::success();
  }
};
} // namespace

namespace {
/// Lower `fir.box_addr` to the sequence of operations to extract the first
/// element of the box.
struct BoxAddrOpConversion : public fir::FIROpConversion<fir::BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxAddrOp boxaddr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxaddr.getLoc();
    if (auto argty =
            mlir::dyn_cast<fir::BaseBoxType>(boxaddr.getVal().getType())) {
      TypePair boxTyPair = getBoxTypePair(argty);
      rewriter.replaceOp(boxaddr,
                         getBaseAddrFromBox(loc, boxTyPair, a, rewriter));
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxaddr, a, 0);
    }
    return mlir::success();
  }
};

/// Convert `!fir.boxchar_len` to  `!llvm.extractvalue` for the 2nd part of the
/// boxchar.
struct BoxCharLenOpConversion : public fir::FIROpConversion<fir::BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
struct BoxDimsOpConversion : public fir::FIROpConversion<fir::BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxDimsOp boxdims, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 3> resultTypes = {
        convertType(boxdims.getResult(0).getType()),
        convertType(boxdims.getResult(1).getType()),
        convertType(boxdims.getResult(2).getType()),
    };
    TypePair boxTyPair = getBoxTypePair(boxdims.getVal().getType());
    auto results = getDimsFromBox(boxdims.getLoc(), resultTypes, boxTyPair,
                                  adaptor.getOperands()[0],
                                  adaptor.getOperands()[1], rewriter);
    rewriter.replaceOp(boxdims, results);
    return mlir::success();
  }
};

/// Lower `fir.box_elesize` to a sequence of operations ro extract the size of
/// an element in the boxed value.
struct BoxEleSizeOpConversion : public fir::FIROpConversion<fir::BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxEleSizeOp boxelesz, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxelesz.getLoc();
    auto ty = convertType(boxelesz.getType());
    TypePair boxTyPair = getBoxTypePair(boxelesz.getVal().getType());
    auto elemSize = getElementSizeFromBox(loc, ty, boxTyPair, box, rewriter);
    rewriter.replaceOp(boxelesz, elemSize);
    return mlir::success();
  }
};

/// Lower `fir.box_isalloc` to a sequence of operations to determine if the
/// boxed value was from an ALLOCATABLE entity.
struct BoxIsAllocOpConversion : public fir::FIROpConversion<fir::BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxIsAllocOp boxisalloc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxisalloc.getLoc();
    TypePair boxTyPair = getBoxTypePair(boxisalloc.getVal().getType());
    mlir::Value check =
        genBoxAttributeCheck(loc, boxTyPair, box, rewriter, kAttrAllocatable);
    rewriter.replaceOp(boxisalloc, check);
    return mlir::success();
  }
};

/// Lower `fir.box_isarray` to a sequence of operations to determine if the
/// boxed is an array.
struct BoxIsArrayOpConversion : public fir::FIROpConversion<fir::BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxIsArrayOp boxisarray, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxisarray.getLoc();
    TypePair boxTyPair = getBoxTypePair(boxisarray.getVal().getType());
    mlir::Value rank = getRankFromBox(loc, boxTyPair, a, rewriter);
    mlir::Value c0 = genConstantIndex(loc, rank.getType(), rewriter, 0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisarray, mlir::LLVM::ICmpPredicate::ne, rank, c0);
    return mlir::success();
  }
};

/// Lower `fir.box_isptr` to a sequence of operations to determined if the
/// boxed value was from a POINTER entity.
struct BoxIsPtrOpConversion : public fir::FIROpConversion<fir::BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxIsPtrOp boxisptr, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = boxisptr.getLoc();
    TypePair boxTyPair = getBoxTypePair(boxisptr.getVal().getType());
    mlir::Value check =
        genBoxAttributeCheck(loc, boxTyPair, box, rewriter, kAttrPointer);
    rewriter.replaceOp(boxisptr, check);
    return mlir::success();
  }
};

/// Lower `fir.box_rank` to the sequence of operation to extract the rank from
/// the box.
struct BoxRankOpConversion : public fir::FIROpConversion<fir::BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxRankOp boxrank, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value a = adaptor.getOperands()[0];
    auto loc = boxrank.getLoc();
    mlir::Type ty = convertType(boxrank.getType());
    TypePair boxTyPair =
        getBoxTypePair(fir::unwrapRefType(boxrank.getBox().getType()));
    mlir::Value rank = getRankFromBox(loc, boxTyPair, a, rewriter);
    mlir::Value result = integerCast(loc, rewriter, ty, rank);
    rewriter.replaceOp(boxrank, result);
    return mlir::success();
  }
};

/// Lower `fir.boxproc_host` operation. Extracts the host pointer from the
/// boxproc.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct BoxProcHostOpConversion
    : public fir::FIROpConversion<fir::BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxProcHostOp boxprochost, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(boxprochost.getLoc(), "fir.boxproc_host codegen");
    return mlir::failure();
  }
};

/// Lower `fir.box_tdesc` to the sequence of operations to extract the type
/// descriptor from the box.
struct BoxTypeDescOpConversion
    : public fir::FIROpConversion<fir::BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxTypeDescOp boxtypedesc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    TypePair boxTyPair = getBoxTypePair(boxtypedesc.getBox().getType());
    auto typeDescAddr =
        loadTypeDescAddress(boxtypedesc.getLoc(), boxTyPair, box, rewriter);
    rewriter.replaceOp(boxtypedesc, typeDescAddr);
    return mlir::success();
  }
};

/// Lower `fir.box_typecode` to a sequence of operations to extract the type
/// code in the boxed value.
struct BoxTypeCodeOpConversion
    : public fir::FIROpConversion<fir::BoxTypeCodeOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxTypeCodeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value box = adaptor.getOperands()[0];
    auto loc = box.getLoc();
    auto ty = convertType(op.getType());
    TypePair boxTyPair = getBoxTypePair(op.getBox().getType());
    auto typeCode =
        getValueFromBox(loc, boxTyPair, box, ty, rewriter, kTypePosInBox);
    rewriter.replaceOp(op, typeCode);
    return mlir::success();
  }
};

/// Lower `fir.string_lit` to LLVM IR dialect operation.
struct StringLitOpConversion : public fir::FIROpConversion<fir::StringLitOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::StringLitOp constop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(constop.getType());
    auto attr = constop.getValue();
    if (mlir::isa<mlir::StringAttr>(attr)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, attr);
      return mlir::success();
    }

    auto charTy = mlir::cast<fir::CharacterType>(constop.getType());
    unsigned bits = lowerTy().characterBitsize(charTy);
    mlir::Type intTy = rewriter.getIntegerType(bits);
    mlir::Location loc = constop.getLoc();
    mlir::Value cst = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    if (auto arr = mlir::dyn_cast<mlir::DenseElementsAttr>(attr)) {
      cst = rewriter.create<mlir::LLVM::ConstantOp>(loc, ty, arr);
    } else if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      for (auto a : llvm::enumerate(arr.getValue())) {
        // convert each character to a precise bitsize
        auto elemAttr = mlir::IntegerAttr::get(
            intTy,
            mlir::cast<mlir::IntegerAttr>(a.value()).getValue().zextOrTrunc(
                bits));
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
struct CallOpConversion : public fir::FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::CallOp call, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    // Convert arith::FastMathFlagsAttr to LLVM::FastMathFlagsAttr.
    mlir::arith::AttrConvertFastMathToLLVM<fir::CallOp, mlir::LLVM::CallOp>
        attrConvert(call);
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        call, resultTys, adaptor.getOperands(),
        addLLVMOpBundleAttrs(rewriter, attrConvert.getAttrs(),
                             adaptor.getOperands().size()));
    return mlir::success();
  }
};
} // namespace

static mlir::Type getComplexEleTy(mlir::Type complex) {
  return mlir::cast<mlir::ComplexType>(complex).getElementType();
}

namespace {
/// Compare complex values
///
/// Per 10.1, the only comparisons available are .EQ. (oeq) and .NE. (une).
///
/// For completeness, all other comparison are done on the real component only.
struct CmpcOpConversion : public fir::FIROpConversion<fir::CmpcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::CmpcOp cmp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::Type resTy = convertType(cmp.getType());
    mlir::Location loc = cmp.getLoc();
    mlir::LLVM::FastmathFlags fmf =
        mlir::arith::convertArithFastMathFlagsToLLVM(cmp.getFastmath());
    mlir::LLVM::FCmpPredicate pred =
        static_cast<mlir::LLVM::FCmpPredicate>(cmp.getPredicate());
    auto rcp = rewriter.create<mlir::LLVM::FCmpOp>(
        loc, resTy, pred,
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[0], 0),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[1], 0), fmf);
    auto icp = rewriter.create<mlir::LLVM::FCmpOp>(
        loc, resTy, pred,
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[0], 1),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, operands[1], 1), fmf);
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

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public fir::FIROpConversion<fir::ConvertOp> {
  using FIROpConversion::FIROpConversion;

  static bool isFloatingPointTy(mlir::Type ty) {
    return mlir::isa<mlir::FloatType>(ty);
  }

  llvm::LogicalResult
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

    if (mlir::isa<fir::RecordType>(toFirTy)) {
      // Convert to compatible BIND(C) record type.
      // Double check that the record types are compatible (it should have
      // already been checked by the verifier).
      assert(mlir::cast<fir::RecordType>(fromFirTy).getTypeList() ==
                 mlir::cast<fir::RecordType>(toFirTy).getTypeList() &&
             "incompatible record types");

      auto toStTy = mlir::cast<mlir::LLVM::LLVMStructType>(toTy);
      mlir::Value val = rewriter.create<mlir::LLVM::UndefOp>(loc, toStTy);
      auto indexTypeMap = toStTy.getSubelementIndexMap();
      assert(indexTypeMap.has_value() && "invalid record type");

      for (auto [attr, type] : indexTypeMap.value()) {
        int64_t index = mlir::cast<mlir::IntegerAttr>(attr).getInt();
        auto extVal =
            rewriter.create<mlir::LLVM::ExtractValueOp>(loc, op0, index);
        val =
            rewriter.create<mlir::LLVM::InsertValueOp>(loc, val, extVal, index);
      }

      rewriter.replaceOp(convert, val);
      return mlir::success();
    }

    if (mlir::isa<fir::LogicalType>(fromFirTy) ||
        mlir::isa<fir::LogicalType>(toFirTy)) {
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
      if (!mlir::isa<mlir::IntegerType>(fromTy) ||
          !mlir::isa<mlir::IntegerType>(toTy))
        return mlir::emitError(loc)
               << "unsupported types for logical conversion: " << fromTy
               << " -> " << toTy;

      // Do folding for constant inputs.
      if (auto constVal = fir::getIntIfConstant(op0)) {
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
      if (mlir::isa<mlir::IntegerType>(toTy)) {
        if (toTy.isUnsignedInteger())
          rewriter.replaceOpWithNewOp<mlir::LLVM::FPToUIOp>(convert, toTy, op0);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (mlir::isa<mlir::IntegerType>(fromTy)) {
      // Integer to integer conversion.
      if (mlir::isa<mlir::IntegerType>(toTy)) {
        auto fromBits = mlir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = mlir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        assert(fromBits != toBits);
        if (fromBits > toBits) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::TruncOp>(convert, toTy, op0);
          return mlir::success();
        }
        if (fromFirTy == i1Type || fromFirTy.isUnsignedInteger()) {
          rewriter.replaceOpWithNewOp<mlir::LLVM::ZExtOp>(convert, toTy, op0);
          return mlir::success();
        }
        rewriter.replaceOpWithNewOp<mlir::LLVM::SExtOp>(convert, toTy, op0);
        return mlir::success();
      }
      // Integer to floating point conversion.
      if (isFloatingPointTy(toTy)) {
        if (fromTy.isUnsignedInteger())
          rewriter.replaceOpWithNewOp<mlir::LLVM::UIToFPOp>(convert, toTy, op0);
        else
          rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(convert, toTy, op0);
        return mlir::success();
      }
      // Integer to pointer conversion.
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(toTy)) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(convert, toTy, op0);
        return mlir::success();
      }
    } else if (mlir::isa<mlir::LLVM::LLVMPointerType>(fromTy)) {
      // Pointer to integer conversion.
      if (mlir::isa<mlir::IntegerType>(toTy)) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::PtrToIntOp>(convert, toTy, op0);
        return mlir::success();
      }
      // Pointer to pointer conversion.
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(toTy)) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(convert, toTy, op0);
        return mlir::success();
      }
    }
    return emitError(loc) << "cannot convert " << fromTy << " to " << toTy;
  }
};

/// `fir.type_info` operation has no specific CodeGen. The operation is
/// only used to carry information during FIR to FIR passes. It may be used
/// in the future to generate the runtime type info data structures instead
/// of generating them in lowering.
struct TypeInfoOpConversion : public fir::FIROpConversion<fir::TypeInfoOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::TypeInfoOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// `fir.dt_entry` operation has no specific CodeGen. The operation is only used
/// to carry information during FIR to FIR passes.
struct DTEntryOpConversion : public fir::FIROpConversion<fir::DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::DTEntryOp op, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/// Lower `fir.global_len` operation.
struct GlobalLenOpConversion : public fir::FIROpConversion<fir::GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::GlobalLenOp globalLen, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(globalLen.getLoc(), "fir.global_len codegen");
    return mlir::failure();
  }
};

/// Lower fir.len_param_index
struct LenParamIndexOpConversion
    : public fir::FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  llvm::LogicalResult
  matchAndRewrite(fir::LenParamIndexOp lenp, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(lenp.getLoc(), "fir.len_param_index codegen");
  }
};

/// Convert `!fir.emboxchar<!fir.char<KIND, ?>, #n>` into a sequence of
/// instructions that generate `!llvm.struct<(ptr<ik>, i64)>`. The 1st element
/// in this struct is a pointer. Its type is determined from `KIND`. The 2nd
/// element is the length of the character buffer (`#n`).
struct EmboxCharOpConversion : public fir::FIROpConversion<fir::EmboxCharOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::EmboxCharOp emboxChar, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();

    mlir::Value charBuffer = operands[0];
    mlir::Value charBufferLen = operands[1];

    mlir::Location loc = emboxChar.getLoc();
    mlir::Type llvmStructTy = convertType(emboxChar.getType());
    auto llvmStruct = rewriter.create<mlir::LLVM::UndefOp>(loc, llvmStructTy);

    mlir::Type lenTy =
        mlir::cast<mlir::LLVM::LLVMStructType>(llvmStructTy).getBody()[1];
    mlir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, charBufferLen);

    mlir::Type addrTy =
        mlir::cast<mlir::LLVM::LLVMStructType>(llvmStructTy).getBody()[0];
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

template <typename ModuleOp>
static mlir::SymbolRefAttr
getMallocInModule(ModuleOp mod, fir::AllocMemOp op,
                  mlir::ConversionPatternRewriter &rewriter) {
  static constexpr char mallocName[] = "malloc";
  if (auto mallocFunc =
          mod.template lookupSymbol<mlir::LLVM::LLVMFuncOp>(mallocName))
    return mlir::SymbolRefAttr::get(mallocFunc);
  if (auto userMalloc =
          mod.template lookupSymbol<mlir::func::FuncOp>(mallocName))
    return mlir::SymbolRefAttr::get(userMalloc);

  mlir::OpBuilder moduleBuilder(mod.getBodyRegion());
  auto indexType = mlir::IntegerType::get(op.getContext(), 64);
  auto mallocDecl = moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      op.getLoc(), mallocName,
      mlir::LLVM::LLVMFunctionType::get(getLlvmPtrType(op.getContext()),
                                        indexType,
                                        /*isVarArg=*/false));
  return mlir::SymbolRefAttr::get(mallocDecl);
}

/// Return the LLVMFuncOp corresponding to the standard malloc call.
static mlir::SymbolRefAttr
getMalloc(fir::AllocMemOp op, mlir::ConversionPatternRewriter &rewriter) {
  if (auto mod = op->getParentOfType<mlir::gpu::GPUModuleOp>())
    return getMallocInModule(mod, op, rewriter);
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  return getMallocInModule(mod, op, rewriter);
}

/// Helper function for generating the LLVM IR that computes the distance
/// in bytes between adjacent elements pointed to by a pointer
/// of type \p ptrTy. The result is returned as a value of \p idxTy integer
/// type.
static mlir::Value
computeElementDistance(mlir::Location loc, mlir::Type llvmObjectType,
                       mlir::Type idxTy,
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
  auto llvmPtrTy = ::getLlvmPtrType(llvmObjectType.getContext());
  auto nullPtr = rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPtrTy);
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(
      loc, llvmPtrTy, llvmObjectType, nullPtr,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{1});
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
  return computeElementDistance(loc, llTy, idxTy, rewriter);
}

namespace {
/// Lower a `fir.allocmem` instruction into `llvm.call @malloc`
struct AllocMemOpConversion : public fir::FIROpConversion<fir::AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AllocMemOp heap, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type heapTy = heap.getType();
    mlir::Location loc = heap.getLoc();
    auto ity = lowerTy().indexType();
    mlir::Type dataTy = fir::unwrapRefType(heapTy);
    mlir::Type llvmObjectTy = convertObjectType(dataTy);
    if (fir::isRecordWithTypeParameters(fir::unwrapSequenceType(dataTy)))
      TODO(loc, "fir.allocmem codegen of derived type with length parameters");
    mlir::Value size = genTypeSizeInBytes(loc, ity, rewriter, llvmObjectTy);
    if (auto scaleSize = genAllocationScaleSize(heap, ity, rewriter))
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, scaleSize);
    for (mlir::Value opnd : adaptor.getOperands())
      size = rewriter.create<mlir::LLVM::MulOp>(
          loc, ity, size, integerCast(loc, rewriter, ity, opnd));
    heap->setAttr("callee", getMalloc(heap, rewriter));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        heap, ::getLlvmPtrType(heap.getContext()), size,
        addLLVMOpBundleAttrs(rewriter, heap->getAttrs(), 1));
    return mlir::success();
  }

  /// Compute the allocation size in bytes of the element type of
  /// \p llTy pointer type. The result is returned as a value of \p idxTy
  /// integer type.
  mlir::Value genTypeSizeInBytes(mlir::Location loc, mlir::Type idxTy,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::Type llTy) const {
    return computeElementDistance(loc, llTy, idxTy, rewriter);
  }
};
} // namespace

/// Return the LLVMFuncOp corresponding to the standard free call.
template <typename ModuleOp>
static mlir::SymbolRefAttr
getFreeInModule(ModuleOp mod, fir::FreeMemOp op,
                mlir::ConversionPatternRewriter &rewriter) {
  static constexpr char freeName[] = "free";
  // Check if free already defined in the module.
  if (auto freeFunc =
          mod.template lookupSymbol<mlir::LLVM::LLVMFuncOp>(freeName))
    return mlir::SymbolRefAttr::get(freeFunc);
  if (auto freeDefinedByUser =
          mod.template lookupSymbol<mlir::func::FuncOp>(freeName))
    return mlir::SymbolRefAttr::get(freeDefinedByUser);
  // Create llvm declaration for free.
  mlir::OpBuilder moduleBuilder(mod.getBodyRegion());
  auto voidType = mlir::LLVM::LLVMVoidType::get(op.getContext());
  auto freeDecl = moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), freeName,
      mlir::LLVM::LLVMFunctionType::get(voidType,
                                        getLlvmPtrType(op.getContext()),
                                        /*isVarArg=*/false));
  return mlir::SymbolRefAttr::get(freeDecl);
}

static mlir::SymbolRefAttr getFree(fir::FreeMemOp op,
                                   mlir::ConversionPatternRewriter &rewriter) {
  if (auto mod = op->getParentOfType<mlir::gpu::GPUModuleOp>())
    return getFreeInModule(mod, op, rewriter);
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  return getFreeInModule(mod, op, rewriter);
}

static unsigned getDimension(mlir::LLVM::LLVMArrayType ty) {
  unsigned result = 1;
  for (auto eleTy =
           mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(ty.getElementType());
       eleTy; eleTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(
                  eleTy.getElementType()))
    ++result;
  return result;
}

namespace {
/// Lower a `fir.freemem` instruction into `llvm.call @free`
struct FreeMemOpConversion : public fir::FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::FreeMemOp freemem, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = freemem.getLoc();
    freemem->setAttr("callee", getFree(freemem, rewriter));
    rewriter.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{}, mlir::ValueRange{adaptor.getHeapref()},
        addLLVMOpBundleAttrs(rewriter, freemem->getAttrs(), 1));
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

static mlir::Value genSourceFile(mlir::Location loc, mlir::ModuleOp mod,
                                 mlir::ConversionPatternRewriter &rewriter) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    auto fn = flc.getFilename().str() + '\0';
    std::string globalName = fir::factory::uniqueCGIdent("cl", fn);

    if (auto g = mod.lookupSymbol<fir::GlobalOp>(globalName)) {
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, g.getName());
    } else if (auto g = mod.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ptrTy, g.getName());
    }

    auto crtInsPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(mod.getBody(), mod.getBody()->end());
    auto arrayTy = mlir::LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(rewriter.getContext(), 8), fn.size());
    mlir::LLVM::GlobalOp globalOp = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, arrayTy, /*constant=*/true, mlir::LLVM::Linkage::Linkonce,
        globalName, mlir::Attribute());

    mlir::Region &region = globalOp.getInitializerRegion();
    mlir::Block *block = rewriter.createBlock(&region);
    rewriter.setInsertionPoint(block, block->begin());
    mlir::Value constValue = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, arrayTy, rewriter.getStringAttr(fn));
    rewriter.create<mlir::LLVM::ReturnOp>(loc, constValue);
    rewriter.restoreInsertionPoint(crtInsPt);
    return rewriter.create<mlir::LLVM::AddressOfOp>(loc, ptrTy,
                                                    globalOp.getName());
  }
  return rewriter.create<mlir::LLVM::ZeroOp>(loc, ptrTy);
}

static mlir::Value genSourceLine(mlir::Location loc,
                                 mlir::ConversionPatternRewriter &rewriter) {
  if (auto flc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(),
                                                   flc.getLine());
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0);
}

static mlir::Value
genCUFAllocDescriptor(mlir::Location loc,
                      mlir::ConversionPatternRewriter &rewriter,
                      mlir::ModuleOp mod, fir::BaseBoxType boxTy,
                      const fir::LLVMTypeConverter &typeConverter) {
  std::optional<mlir::DataLayout> dl =
      fir::support::getOrSetMLIRDataLayout(mod, /*allowDefaultLayout=*/true);
  if (!dl)
    mlir::emitError(mod.getLoc(),
                    "module operation must carry a data layout attribute "
                    "to generate llvm IR from FIR");

  mlir::Value sourceFile = genSourceFile(loc, mod, rewriter);
  mlir::Value sourceLine = genSourceLine(loc, rewriter);

  mlir::MLIRContext *ctx = mod.getContext();

  mlir::LLVM::LLVMPointerType llvmPointerType =
      mlir::LLVM::LLVMPointerType::get(ctx);
  mlir::Type llvmInt32Type = mlir::IntegerType::get(ctx, 32);
  mlir::Type llvmIntPtrType =
      mlir::IntegerType::get(ctx, typeConverter.getPointerBitwidth(0));
  auto fctTy = mlir::LLVM::LLVMFunctionType::get(
      llvmPointerType, {llvmIntPtrType, llvmPointerType, llvmInt32Type});

  auto llvmFunc = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      RTNAME_STRING(CUFAllocDescriptor));
  auto funcFunc =
      mod.lookupSymbol<mlir::func::FuncOp>(RTNAME_STRING(CUFAllocDescriptor));
  if (!llvmFunc && !funcFunc)
    mlir::OpBuilder::atBlockEnd(mod.getBody())
        .create<mlir::LLVM::LLVMFuncOp>(loc, RTNAME_STRING(CUFAllocDescriptor),
                                        fctTy);

  mlir::Type structTy = typeConverter.convertBoxTypeAsStruct(boxTy);
  std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
  mlir::Value sizeInBytes =
      genConstantIndex(loc, llvmIntPtrType, rewriter, boxSize);
  llvm::SmallVector args = {sizeInBytes, sourceFile, sourceLine};
  return rewriter
      .create<mlir::LLVM::CallOp>(loc, fctTy, RTNAME_STRING(CUFAllocDescriptor),
                                  args)
      .getResult();
}

/// Common base class for embox to descriptor conversion.
template <typename OP>
struct EmboxCommonConversion : public fir::FIROpConversion<OP> {
  using fir::FIROpConversion<OP>::FIROpConversion;
  using TypePair = typename fir::FIROpConversion<OP>::TypePair;

  static int getCFIAttr(fir::BaseBoxType boxTy) {
    auto eleTy = boxTy.getEleTy();
    if (mlir::isa<fir::PointerType>(eleTy))
      return CFI_attribute_pointer;
    if (mlir::isa<fir::HeapType>(eleTy))
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
    auto len64 = fir::FIROpConversion<OP>::integerCast(loc, rewriter, i64Ty,
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
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(boxEleTy))
      return getSizeAndTypeCode(loc, rewriter, seqTy.getEleTy(), lenParams);
    if (mlir::isa<mlir::NoneType>(
            boxEleTy)) // unlimited polymorphic or assumed type
      return {rewriter.create<mlir::LLVM::ConstantOp>(loc, i64Ty, 0),
              this->genConstantOffset(loc, rewriter, CFI_type_other)};
    mlir::Value typeCodeVal = this->genConstantOffset(
        loc, rewriter,
        fir::getTypeCode(boxEleTy, this->lowerTy().getKindMap()));
    if (fir::isa_integer(boxEleTy) ||
        mlir::dyn_cast<fir::LogicalType>(boxEleTy) || fir::isa_real(boxEleTy) ||
        fir::isa_complex(boxEleTy))
      return {genTypeStrideInBytes(loc, i64Ty, rewriter,
                                   this->convertType(boxEleTy)),
              typeCodeVal};
    if (auto charTy = mlir::dyn_cast<fir::CharacterType>(boxEleTy))
      return {getCharacterByteSize(loc, rewriter, charTy, lenParams),
              typeCodeVal};
    if (fir::isa_ref_type(boxEleTy)) {
      auto ptrTy = ::getLlvmPtrType(rewriter.getContext());
      return {genTypeStrideInBytes(loc, i64Ty, rewriter, ptrTy), typeCodeVal};
    }
    if (mlir::isa<fir::RecordType>(boxEleTy))
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
    if (!bitcast)
      value = this->integerCast(loc, rewriter, fldTy, value);
    // bitcast are no-ops with LLVM opaque pointers.
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
  template <typename ModOpTy>
  mlir::Value
  getTypeDescriptor(ModOpTy mod, mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, fir::RecordType recType) const {
    std::string name =
        this->options.typeDescriptorsRenamedForAssembly
            ? fir::NameUniquer::getTypeDescriptorAssemblyName(recType.getName())
            : fir::NameUniquer::getTypeDescriptorName(recType.getName());
    mlir::Type llvmPtrTy = ::getLlvmPtrType(mod.getContext());
    if (auto global = mod.template lookupSymbol<fir::GlobalOp>(name)) {
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, llvmPtrTy,
                                                      global.getSymName());
    }
    if (auto global = mod.template lookupSymbol<mlir::LLVM::GlobalOp>(name)) {
      // The global may have already been translated to LLVM.
      return rewriter.create<mlir::LLVM::AddressOfOp>(loc, llvmPtrTy,
                                                      global.getSymName());
    }
    // Type info derived types do not have type descriptors since they are the
    // types defining type descriptors.
    if (!this->options.ignoreMissingTypeDescriptors &&
        !fir::NameUniquer::belongsToModule(
            name, Fortran::semantics::typeInfoBuiltinModule))
      fir::emitFatalError(
          loc, "runtime derived type info descriptor was not generated");
    return rewriter.create<mlir::LLVM::ZeroOp>(loc, llvmPtrTy);
  }

  template <typename ModOpTy>
  mlir::Value populateDescriptor(mlir::Location loc, ModOpTy mod,
                                 fir::BaseBoxType boxTy, mlir::Type inputType,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 unsigned rank, mlir::Value eleSize,
                                 mlir::Value cfiTy, mlir::Value typeDesc,
                                 int allocatorIdx = kDefaultAllocator,
                                 mlir::Value extraField = {}) const {
    auto llvmBoxTy = this->lowerTy().convertBoxTypeAsStruct(boxTy, rank);
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

    if (extraField) {
      // Make sure to set the addendum presence flag according to the
      // destination box.
      if (hasAddendum) {
        auto maskAttr = mlir::IntegerAttr::get(
            rewriter.getIntegerType(8, /*isSigned=*/false),
            llvm::APInt(8, (uint64_t)_CFI_ADDENDUM_FLAG, /*isSigned=*/false));
        mlir::LLVM::ConstantOp mask = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, rewriter.getI8Type(), maskAttr);
        extraField = rewriter.create<mlir::LLVM::OrOp>(loc, extraField, mask);
      } else {
        auto maskAttr = mlir::IntegerAttr::get(
            rewriter.getIntegerType(8, /*isSigned=*/false),
            llvm::APInt(8, (uint64_t)~_CFI_ADDENDUM_FLAG, /*isSigned=*/true));
        mlir::LLVM::ConstantOp mask = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, rewriter.getI8Type(), maskAttr);
        extraField = rewriter.create<mlir::LLVM::AndOp>(loc, extraField, mask);
      }
      // Extra field value is provided so just use it.
      descriptor =
          insertField(rewriter, loc, descriptor, {kExtraPosInBox}, extraField);
    } else {
      // Compute the value of the extra field based on allocator_idx and
      // addendum present.
      unsigned extra = allocatorIdx << _CFI_ALLOCATOR_IDX_SHIFT;
      if (hasAddendum)
        extra |= _CFI_ADDENDUM_FLAG;
      descriptor = insertField(rewriter, loc, descriptor, {kExtraPosInBox},
                               this->genI32Constant(loc, rewriter, extra));
    }

    if (hasAddendum) {
      unsigned typeDescFieldId = getTypeDescFieldId(boxTy);
      if (!typeDesc) {
        if (useInputType) {
          mlir::Type innerType = fir::unwrapInnerType(inputType);
          if (innerType && mlir::isa<fir::RecordType>(innerType)) {
            auto recTy = mlir::dyn_cast<fir::RecordType>(innerType);
            typeDesc = getTypeDescriptor(mod, rewriter, loc, recTy);
          } else {
            // Unlimited polymorphic type descriptor with no record type. Set
            // type descriptor address to a clean state.
            typeDesc = rewriter.create<mlir::LLVM::ZeroOp>(
                loc, ::getLlvmPtrType(mod.getContext()));
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
      // Always initialize the length parameter field to zero to avoid issues
      // with uninitialized values in Fortran code trying to compare physical
      // representation of derived types with pointer/allocatable components.
      // This has been seen in hashing algorithms using TRANSFER.
      mlir::Value zero =
          genConstantIndex(loc, rewriter.getI64Type(), rewriter, 0);
      descriptor = insertField(rewriter, loc, descriptor,
                               {getLenParamFieldId(boxTy), 0}, zero);
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
    auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(box.getType());
    bool useInputType = fir::isPolymorphicType(boxTy) &&
                        !fir::isUnlimitedPolymorphicType(inputType);
    llvm::SmallVector<mlir::Value> typeparams = lenParams;
    if constexpr (!std::is_same_v<BOX, fir::EmboxOp>) {
      if (!box.getSubstr().empty() && fir::hasDynamicSize(boxTy.getEleTy()))
        typeparams.push_back(substrParams[1]);
    }

    int allocatorIdx = 0;
    if constexpr (std::is_same_v<BOX, fir::EmboxOp> ||
                  std::is_same_v<BOX, fir::cg::XEmboxOp>) {
      if (box.getAllocatorIdx())
        allocatorIdx = *box.getAllocatorIdx();
    }

    // Write each of the fields with the appropriate values.
    // When emboxing an element to a polymorphic descriptor, use the
    // input type since the destination descriptor type has not the exact
    // information.
    auto [eleSize, cfiTy] = getSizeAndTypeCode(
        loc, rewriter, useInputType ? inputType : boxTy.getEleTy(), typeparams);

    mlir::Value typeDesc;
    mlir::Value extraField;
    // When emboxing to a polymorphic box, get the type descriptor, type code
    // and element size from the source box if any.
    if (fir::isPolymorphicType(boxTy) && sourceBox) {
      TypePair sourceBoxTyPair = this->getBoxTypePair(sourceBoxType);
      typeDesc =
          this->loadTypeDescAddress(loc, sourceBoxTyPair, sourceBox, rewriter);
      mlir::Type idxTy = this->lowerTy().indexType();
      eleSize = this->getElementSizeFromBox(loc, idxTy, sourceBoxTyPair,
                                            sourceBox, rewriter);
      cfiTy = this->getValueFromBox(loc, sourceBoxTyPair, sourceBox,
                                    cfiTy.getType(), rewriter, kTypePosInBox);
      extraField =
          this->getExtraFromBox(loc, sourceBoxTyPair, sourceBox, rewriter);
    }

    mlir::Value descriptor;
    if (auto gpuMod = box->template getParentOfType<mlir::gpu::GPUModuleOp>())
      descriptor = populateDescriptor(loc, gpuMod, boxTy, inputType, rewriter,
                                      rank, eleSize, cfiTy, typeDesc,
                                      allocatorIdx, extraField);
    else if (auto mod = box->template getParentOfType<mlir::ModuleOp>())
      descriptor = populateDescriptor(loc, mod, boxTy, inputType, rewriter,
                                      rank, eleSize, cfiTy, typeDesc,
                                      allocatorIdx, extraField);

    return {boxTy, descriptor, eleSize};
  }

  std::tuple<fir::BaseBoxType, mlir::Value, mlir::Value>
  consDescriptorPrefix(fir::cg::XReboxOp box, mlir::Value loweredBox,
                       mlir::ConversionPatternRewriter &rewriter, unsigned rank,
                       mlir::ValueRange substrParams,
                       mlir::ValueRange lenParams,
                       mlir::Value typeDesc = {}) const {
    auto loc = box.getLoc();
    auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(box.getType());
    auto inputBoxTy = mlir::dyn_cast<fir::BaseBoxType>(box.getBox().getType());
    auto inputBoxTyPair = this->getBoxTypePair(inputBoxTy);
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
      eleSize = this->getElementSizeFromBox(loc, idxTy, inputBoxTyPair,
                                            loweredBox, rewriter);
      cfiTy = this->getValueFromBox(loc, inputBoxTyPair, loweredBox,
                                    cfiTy.getType(), rewriter, kTypePosInBox);
      // TODO: For initial box that are unlimited polymorphic entities, this
      // code must be made conditional because unlimited polymorphic entities
      // with intrinsic type spec does not have addendum.
      if (fir::boxHasAddendum(inputBoxTy))
        typeDesc = this->loadTypeDescAddress(loc, inputBoxTyPair, loweredBox,
                                             rewriter);
    }

    mlir::Value extraField =
        this->getExtraFromBox(loc, inputBoxTyPair, loweredBox, rewriter);

    mlir::Value descriptor;
    if (auto gpuMod = box->template getParentOfType<mlir::gpu::GPUModuleOp>())
      descriptor =
          populateDescriptor(loc, gpuMod, boxTy, box.getBox().getType(),
                             rewriter, rank, eleSize, cfiTy, typeDesc,
                             /*allocatorIdx=*/kDefaultAllocator, extraField);
    else if (auto mod = box->template getParentOfType<mlir::ModuleOp>())
      descriptor =
          populateDescriptor(loc, mod, boxTy, box.getBox().getType(), rewriter,
                             rank, eleSize, cfiTy, typeDesc,
                             /*allocatorIdx=*/kDefaultAllocator, extraField);

    return {boxTy, descriptor, eleSize};
  }

  // Compute the base address of a fir.box given the indices from the slice.
  // The indices from the "outer" dimensions (every dimension after the first
  // one (included) that is not a compile time constant) must have been
  // multiplied with the related extents and added together into \p outerOffset.
  mlir::Value
  genBoxOffsetGep(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  mlir::Value base, mlir::Type llvmBaseObjectType,
                  mlir::Value outerOffset, mlir::ValueRange cstInteriorIndices,
                  mlir::ValueRange componentIndices,
                  std::optional<mlir::Value> substringOffset) const {
    llvm::SmallVector<mlir::LLVM::GEPArg> gepArgs{outerOffset};
    mlir::Type resultTy = llvmBaseObjectType;
    // Fortran is column major, llvm GEP is row major: reverse the indices here.
    for (mlir::Value interiorIndex : llvm::reverse(cstInteriorIndices)) {
      auto arrayTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(resultTy);
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
      if (auto arrayTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(resultTy)) {
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
        mlir::Type outterOffsetTy =
            llvm::cast<mlir::Value>(gepArgs[0]).getType();
        mlir::Value cast =
            this->integerCast(loc, rewriter, outterOffsetTy, *substringOffset);

        gepArgs[0] = rewriter.create<mlir::LLVM::AddOp>(
            loc, outterOffsetTy, llvm::cast<mlir::Value>(gepArgs[0]), cast);
      }
    }
    mlir::Type llvmPtrTy = ::getLlvmPtrType(resultTy.getContext());
    return rewriter.create<mlir::LLVM::GEPOp>(
        loc, llvmPtrTy, llvmBaseObjectType, base, gepArgs);
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
    indices.append(operands.begin() + xbox.getSubcomponentOperandIndex(),
                   operands.begin() + xbox.getSubcomponentOperandIndex() +
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
                               mlir::Value boxValue,
                               bool needDeviceAllocation = false) const {
    if (isInGlobalOp(rewriter))
      return boxValue;
    mlir::Type llvmBoxTy = boxValue.getType();
    mlir::Value storage;
    if (needDeviceAllocation) {
      auto mod = boxValue.getDefiningOp()->getParentOfType<mlir::ModuleOp>();
      auto baseBoxTy = mlir::dyn_cast<fir::BaseBoxType>(boxTy);
      storage =
          genCUFAllocDescriptor(loc, rewriter, mod, baseBoxTy, this->lowerTy());
    } else {
      storage = this->genAllocaAndAddrCastWithType(loc, llvmBoxTy, defaultAlign,
                                                   rewriter);
    }
    auto storeOp = rewriter.create<mlir::LLVM::StoreOp>(loc, boxValue, storage);
    this->attachTBAATag(storeOp, boxTy, boxTy, nullptr);
    return storage;
  }

  /// Compute the extent of a triplet slice (lb:ub:step).
  mlir::Value computeTripletExtent(mlir::ConversionPatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Value lb,
                                   mlir::Value ub, mlir::Value step,
                                   mlir::Value zero, mlir::Type type) const {
    lb = this->integerCast(loc, rewriter, type, lb);
    ub = this->integerCast(loc, rewriter, type, ub);
    step = this->integerCast(loc, rewriter, type, step);
    zero = this->integerCast(loc, rewriter, type, zero);
    mlir::Value extent = rewriter.create<mlir::LLVM::SubOp>(loc, type, ub, lb);
    extent = rewriter.create<mlir::LLVM::AddOp>(loc, type, extent, step);
    extent = rewriter.create<mlir::LLVM::SDivOp>(loc, type, extent, step);
    // If the resulting extent is negative (`ub-lb` and `step` have different
    // signs), zero must be returned instead.
    auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::sgt, extent, zero);
    return rewriter.create<mlir::LLVM::SelectOp>(loc, cmp, extent, zero);
  }
};

/// Create a generic box on a memory reference. This conversions lowers the
/// abstract box to the appropriate, initialized descriptor.
struct EmboxOpConversion : public EmboxCommonConversion<fir::EmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::EmboxOp embox, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::Value sourceBox;
    mlir::Type sourceBoxType;
    if (embox.getSourceBox()) {
      sourceBox = operands[embox.getSourceBoxOperandIndex()];
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

static bool isDeviceAllocation(mlir::Value val, mlir::Value adaptorVal) {
  if (auto loadOp = mlir::dyn_cast_or_null<fir::LoadOp>(val.getDefiningOp()))
    return isDeviceAllocation(loadOp.getMemref(), {});
  if (auto boxAddrOp =
          mlir::dyn_cast_or_null<fir::BoxAddrOp>(val.getDefiningOp()))
    return isDeviceAllocation(boxAddrOp.getVal(), {});
  if (auto convertOp =
          mlir::dyn_cast_or_null<fir::ConvertOp>(val.getDefiningOp()))
    return isDeviceAllocation(convertOp.getValue(), {});
  if (!val.getDefiningOp() && adaptorVal) {
    if (auto blockArg = llvm::cast<mlir::BlockArgument>(adaptorVal)) {
      if (blockArg.getOwner() && blockArg.getOwner()->getParentOp() &&
          blockArg.getOwner()->isEntryBlock()) {
        if (auto func = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(
                *blockArg.getOwner()->getParentOp())) {
          auto argAttrs = func.getArgAttrs(blockArg.getArgNumber());
          for (auto attr : argAttrs) {
            if (attr.getName().getValue().ends_with(cuf::getDataAttrName())) {
              auto dataAttr =
                  mlir::dyn_cast<cuf::DataAttributeAttr>(attr.getValue());
              if (dataAttr.getValue() != cuf::DataAttribute::Pinned &&
                  dataAttr.getValue() != cuf::DataAttribute::Unified)
                return true;
            }
          }
        }
      }
    }
  }
  if (auto callOp = mlir::dyn_cast_or_null<fir::CallOp>(val.getDefiningOp()))
    if (callOp.getCallee() &&
        (callOp.getCallee().value().getRootReference().getValue().starts_with(
             RTNAME_STRING(CUFMemAlloc)) ||
         callOp.getCallee().value().getRootReference().getValue().starts_with(
             RTNAME_STRING(CUFAllocDescriptor))))
      return true;
  return false;
}

/// Create a generic box on a memory reference.
struct XEmboxOpConversion : public EmboxCommonConversion<fir::cg::XEmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::cg::XEmboxOp xbox, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::Value sourceBox;
    mlir::Type sourceBoxType;
    if (xbox.getSourceBox()) {
      sourceBox = operands[xbox.getSourceBoxOperandIndex()];
      sourceBoxType = xbox.getSourceBox().getType();
    }
    auto [boxTy, dest, resultEleSize] = consDescriptorPrefix(
        xbox, fir::unwrapRefType(xbox.getMemref().getType()), rewriter,
        xbox.getOutRank(), adaptor.getSubstr(), adaptor.getLenParams(),
        sourceBox, sourceBoxType);
    // Generate the triples in the dims field of the descriptor
    auto i64Ty = mlir::IntegerType::get(xbox.getContext(), 64);
    assert(!xbox.getShape().empty() && "must have a shape");
    unsigned shapeOffset = xbox.getShapeOperandIndex();
    bool hasShift = !xbox.getShift().empty();
    unsigned shiftOffset = xbox.getShiftOperandIndex();
    bool hasSlice = !xbox.getSlice().empty();
    unsigned sliceOffset = xbox.getSliceOperandIndex();
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
    assert(mlir::isa<fir::SequenceType>(memEleTy));
    auto seqTy = mlir::cast<fir::SequenceType>(memEleTy);
    mlir::Type seqEleTy = seqTy.getEleTy();
    // Adjust the element scaling factor if the element is a dependent type.
    if (fir::hasDynamicSize(seqEleTy)) {
      if (auto charTy = mlir::dyn_cast<fir::CharacterType>(seqEleTy)) {
        // The GEP pointer type decays to llvm.ptr<i[width]>.
        // The scaling factor is the runtime value of the length.
        assert(!adaptor.getLenParams().empty());
        prevPtrOff = FIROpConversion::integerCast(
            loc, rewriter, i64Ty, adaptor.getLenParams().back());
      } else if (mlir::isa<fir::RecordType>(seqEleTy)) {
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
      auto charTy = mlir::cast<fir::CharacterType>(seqEleTy);
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
      mlir::Value extent =
          integerCast(loc, rewriter, i64Ty, operands[shapeOffset]);
      mlir::Value outerExtent = extent;
      bool skipNext = false;
      if (hasSlice) {
        mlir::Value off =
            integerCast(loc, rewriter, i64Ty, operands[sliceOffset]);
        mlir::Value adj = one;
        if (hasShift)
          adj = integerCast(loc, rewriter, i64Ty, operands[shiftOffset]);
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
            mlir::isa<fir::PointerType, fir::HeapType>(eleTy);
        // Lower bound is defaults to 1 for POINTER, ALLOCATABLE, and
        // denormalized descriptors.
        if (isaPointerOrAllocatable || !normalizedLowerBound(xbox))
          lb = one;
        // If there is a shifted origin, and no fir.slice, and this is not
        // a normalized descriptor then use the value from the shift op as
        // the lower bound.
        if (hasShift && !(hasSlice || hasSubcomp || hasSubstr) &&
            (isaPointerOrAllocatable || !normalizedLowerBound(xbox))) {
          lb = integerCast(loc, rewriter, i64Ty, operands[shiftOffset]);
          auto extentIsEmpty = rewriter.create<mlir::LLVM::ICmpOp>(
              loc, mlir::LLVM::ICmpPredicate::eq, extent, zero);
          lb = rewriter.create<mlir::LLVM::SelectOp>(loc, extentIsEmpty, one,
                                                     lb);
        }
        dest = insertLowerBound(rewriter, loc, dest, descIdx, lb);

        dest = insertExtent(rewriter, loc, dest, descIdx, extent);

        // store step (scaled by shaped extent)
        mlir::Value step = prevDimByteStride;
        if (hasSlice) {
          mlir::Value sliceStep =
              integerCast(loc, rewriter, i64Ty, operands[sliceOffset + 2]);
          step =
              rewriter.create<mlir::LLVM::MulOp>(loc, i64Ty, step, sliceStep);
        }
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
    mlir::Value base = adaptor.getMemref();
    if (hasSlice || hasSubcomp || hasSubstr) {
      // Shift the base address.
      llvm::SmallVector<mlir::Value> fieldIndices;
      std::optional<mlir::Value> substringOffset;
      if (hasSubcomp)
        getSubcomponentIndices(xbox, xbox.getMemref(), operands, fieldIndices);
      if (hasSubstr)
        substringOffset = operands[xbox.getSubstrOperandIndex()];
      mlir::Type llvmBaseType =
          convertType(fir::unwrapRefType(xbox.getMemref().getType()));
      base = genBoxOffsetGep(rewriter, loc, base, llvmBaseType, ptrOffset,
                             cstInteriorIndices, fieldIndices, substringOffset);
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    if (fir::isDerivedTypeWithLenParams(boxTy))
      TODO(loc, "fir.embox codegen of derived with length parameters");
    mlir::Value result = placeInMemoryIfNotGlobalInit(
        rewriter, loc, boxTy, dest,
        isDeviceAllocation(xbox.getMemref(), adaptor.getMemref()));
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

  llvm::LogicalResult
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

    TypePair inputBoxTyPair = getBoxTypePair(rebox.getBox().getType());

    // Create new descriptor and fill its non-shape related data.
    llvm::SmallVector<mlir::Value, 2> lenParams;
    mlir::Type inputEleTy = getInputEleTy(rebox);
    if (auto charTy = mlir::dyn_cast<fir::CharacterType>(inputEleTy)) {
      if (charTy.hasConstantLen()) {
        mlir::Value len =
            genConstantIndex(loc, idxTy, rewriter, charTy.getLen());
        lenParams.emplace_back(len);
      } else {
        mlir::Value len = getElementSizeFromBox(loc, idxTy, inputBoxTyPair,
                                                loweredBox, rewriter);
        if (charTy.getFKind() != 1) {
          assert(!isInGlobalOp(rewriter) &&
                 "character target in global op must have constant length");
          mlir::Value width =
              genConstantIndex(loc, idxTy, rewriter, charTy.getFKind());
          len = rewriter.create<mlir::LLVM::SDivOp>(loc, idxTy, len, width);
        }
        lenParams.emplace_back(len);
      }
    } else if (auto recTy = mlir::dyn_cast<fir::RecordType>(inputEleTy)) {
      if (recTy.getNumLenParams() != 0)
        TODO(loc, "reboxing descriptor of derived type with length parameters");
    }

    // Rebox on polymorphic entities needs to carry over the dynamic type.
    mlir::Value typeDescAddr;
    if (mlir::isa<fir::ClassType>(inputBoxTyPair.fir) &&
        mlir::isa<fir::ClassType>(rebox.getType()))
      typeDescAddr =
          loadTypeDescAddress(loc, inputBoxTyPair, loweredBox, rewriter);

    auto [boxTy, dest, eleSize] =
        consDescriptorPrefix(rebox, loweredBox, rewriter, rebox.getOutRank(),
                             adaptor.getSubstr(), lenParams, typeDescAddr);

    // Read input extents, strides, and base address
    llvm::SmallVector<mlir::Value> inputExtents;
    llvm::SmallVector<mlir::Value> inputStrides;
    const unsigned inputRank = rebox.getRank();
    for (unsigned dim = 0; dim < inputRank; ++dim) {
      llvm::SmallVector<mlir::Value, 3> dimInfo =
          getDimsFromBox(loc, {idxTy, idxTy, idxTy}, inputBoxTyPair, loweredBox,
                         dim, rewriter);
      inputExtents.emplace_back(dimInfo[1]);
      inputStrides.emplace_back(dimInfo[2]);
    }

    mlir::Value baseAddr =
        getBaseAddrFromBox(loc, inputBoxTyPair, loweredBox, rewriter);

    if (!rebox.getSlice().empty() || !rebox.getSubcomponent().empty())
      return sliceBox(rebox, adaptor, boxTy, dest, baseAddr, inputExtents,
                      inputStrides, operands, rewriter);
    return reshapeBox(rebox, adaptor, boxTy, dest, baseAddr, inputExtents,
                      inputStrides, operands, rewriter);
  }

private:
  /// Write resulting shape and base address in descriptor, and replace rebox
  /// op.
  llvm::LogicalResult
  finalizeRebox(fir::cg::XReboxOp rebox, OpAdaptor adaptor,
                mlir::Type destBoxTy, mlir::Value dest, mlir::Value base,
                mlir::ValueRange lbounds, mlir::ValueRange extents,
                mlir::ValueRange strides,
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
    mlir::Value result = placeInMemoryIfNotGlobalInit(
        rewriter, rebox.getLoc(), destBoxTy, dest,
        isDeviceAllocation(rebox.getBox(), adaptor.getBox()));
    rewriter.replaceOp(rebox, result);
    return mlir::success();
  }

  // Apply slice given the base address, extents and strides of the input box.
  llvm::LogicalResult
  sliceBox(fir::cg::XReboxOp rebox, OpAdaptor adaptor, mlir::Type destBoxTy,
           mlir::Value dest, mlir::Value base, mlir::ValueRange inputExtents,
           mlir::ValueRange inputStrides, mlir::ValueRange operands,
           mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Location loc = rebox.getLoc();
    mlir::Type byteTy = ::getI8Type(rebox.getContext());
    mlir::Type idxTy = lowerTy().indexType();
    mlir::Value zero = genConstantIndex(loc, idxTy, rewriter, 0);
    // Apply subcomponent and substring shift on base address.
    if (!rebox.getSubcomponent().empty() || !rebox.getSubstr().empty()) {
      // Cast to inputEleTy* so that a GEP can be used.
      mlir::Type inputEleTy = getInputEleTy(rebox);
      mlir::Type llvmBaseObjectType = convertType(inputEleTy);
      llvm::SmallVector<mlir::Value> fieldIndices;
      std::optional<mlir::Value> substringOffset;
      if (!rebox.getSubcomponent().empty())
        getSubcomponentIndices(rebox, rebox.getBox(), operands, fieldIndices);
      if (!rebox.getSubstr().empty())
        substringOffset = operands[rebox.getSubstrOperandIndex()];
      base = genBoxOffsetGep(rewriter, loc, base, llvmBaseObjectType, zero,
                             /*cstInteriorIndices=*/std::nullopt, fieldIndices,
                             substringOffset);
    }

    if (rebox.getSlice().empty())
      // The array section is of the form array[%component][substring], keep
      // the input array extents and strides.
      return finalizeRebox(rebox, adaptor, destBoxTy, dest, base,
                           /*lbounds*/ std::nullopt, inputExtents, inputStrides,
                           rewriter);

    // The slice is of the form array(i:j:k)[%component]. Compute new extents
    // and strides.
    llvm::SmallVector<mlir::Value> slicedExtents;
    llvm::SmallVector<mlir::Value> slicedStrides;
    mlir::Value one = genConstantIndex(loc, idxTy, rewriter, 1);
    const bool sliceHasOrigins = !rebox.getShift().empty();
    unsigned sliceOps = rebox.getSliceOperandIndex();
    unsigned shiftOps = rebox.getShiftOperandIndex();
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
      // Strides from the fir.box are in bytes.
      base = genGEP(loc, byteTy, rewriter, base, offset);
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
    return finalizeRebox(rebox, adaptor, destBoxTy, dest, base,
                         /*lbounds*/ std::nullopt, slicedExtents, slicedStrides,
                         rewriter);
  }

  /// Apply a new shape to the data described by a box given the base address,
  /// extents and strides of the box.
  llvm::LogicalResult
  reshapeBox(fir::cg::XReboxOp rebox, OpAdaptor adaptor, mlir::Type destBoxTy,
             mlir::Value dest, mlir::Value base, mlir::ValueRange inputExtents,
             mlir::ValueRange inputStrides, mlir::ValueRange operands,
             mlir::ConversionPatternRewriter &rewriter) const {
    mlir::ValueRange reboxShifts{
        operands.begin() + rebox.getShiftOperandIndex(),
        operands.begin() + rebox.getShiftOperandIndex() +
            rebox.getShift().size()};
    if (rebox.getShape().empty()) {
      // Only setting new lower bounds.
      return finalizeRebox(rebox, adaptor, destBoxTy, dest, base, reboxShifts,
                           inputExtents, inputStrides, rewriter);
    }

    mlir::Location loc = rebox.getLoc();

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
      mlir::Value rawExtent = operands[rebox.getShapeOperandIndex() + i];
      mlir::Value extent = integerCast(loc, rewriter, idxTy, rawExtent);
      newExtents.emplace_back(extent);
      newStrides.emplace_back(stride);
      // nextStride = extent * stride;
      stride = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, extent, stride);
    }
    return finalizeRebox(rebox, adaptor, destBoxTy, dest, base, reboxShifts,
                         newExtents, newStrides, rewriter);
  }

  /// Return scalar element type of the input box.
  static mlir::Type getInputEleTy(fir::cg::XReboxOp rebox) {
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(rebox.getBox().getType());
    if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty))
      return seqTy.getEleTy();
    return ty;
  }
};

/// Lower `fir.emboxproc` operation. Creates a procedure box.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct EmboxProcOpConversion : public fir::FIROpConversion<fir::EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
      if (auto seq = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(ty)) {
        const auto dim = getDimension(seq);
        if (dim > 1) {
          auto ub = std::min(i + dim, end);
          std::reverse(indices.begin() + i, indices.begin() + ub);
          i += dim - 1;
        }
        ty = getArrayElementType(seq);
      } else if (auto st = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(ty)) {
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
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(*i)) {
        indices.push_back(intAttr.getInt());
      } else {
        auto fieldName = mlir::cast<mlir::StringAttr>(*i).getValue();
        ++i;
        auto ty = mlir::cast<mlir::TypeAttr>(*i).getValue();
        auto index = mlir::cast<fir::RecordType>(ty).getFieldIndex(fieldName);
        indices.push_back(index);
      }
    }
    return indices;
  }

private:
  static mlir::Type getArrayElementType(mlir::LLVM::LLVMArrayType ty) {
    auto eleTy = ty.getElementType();
    while (auto arrTy = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(eleTy))
      eleTy = arrTy.getElementType();
    return eleTy;
  }
};

namespace {
/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion
    : public fir::FIROpAndTypeConversion<fir::ExtractValueOp>,
      public ValueOpCommon {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  llvm::LogicalResult
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
    : public mlir::OpConversionPattern<fir::InsertValueOp>,
      public ValueOpCommon {
  using OpConversionPattern::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(fir::InsertValueOp insertVal, OpAdaptor adaptor,
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
    : public fir::FIROpAndTypeConversion<fir::InsertOnRangeOp> {
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

  llvm::LogicalResult
  doRewrite(fir::InsertOnRangeOp range, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<std::int64_t> dims;
    auto type = adaptor.getOperands()[0].getType();

    // Iteratively extract the array dimensions from the type.
    while (auto t = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
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
    : public fir::FIROpAndTypeConversion<fir::cg::XArrayCoorOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  llvm::LogicalResult
  doRewrite(fir::cg::XArrayCoorOp coor, mlir::Type llvmPtrTy, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    mlir::ValueRange operands = adaptor.getOperands();
    unsigned rank = coor.getRank();
    assert(coor.getIndices().size() == rank);
    assert(coor.getShape().empty() || coor.getShape().size() == rank);
    assert(coor.getShift().empty() || coor.getShift().size() == rank);
    assert(coor.getSlice().empty() || coor.getSlice().size() == 3 * rank);
    mlir::Type idxTy = lowerTy().indexType();
    unsigned indexOffset = coor.getIndicesOperandIndex();
    unsigned shapeOffset = coor.getShapeOperandIndex();
    unsigned shiftOffset = coor.getShiftOperandIndex();
    unsigned sliceOffset = coor.getSliceOperandIndex();
    auto sliceOps = coor.getSlice().begin();
    mlir::Value one = genConstantIndex(loc, idxTy, rewriter, 1);
    mlir::Value prevExt = one;
    mlir::Value offset = genConstantIndex(loc, idxTy, rewriter, 0);
    const bool isShifted = !coor.getShift().empty();
    const bool isSliced = !coor.getSlice().empty();
    const bool baseIsBoxed =
        mlir::isa<fir::BaseBoxType>(coor.getMemref().getType());
    TypePair baseBoxTyPair =
        baseIsBoxed ? getBoxTypePair(coor.getMemref().getType()) : TypePair{};
    mlir::LLVM::IntegerOverflowFlags nsw =
        mlir::LLVM::IntegerOverflowFlags::nsw;

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
      auto idx = rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, index, lb, nsw);
      mlir::Value diff =
          rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, idx, step, nsw);
      if (normalSlice) {
        mlir::Value sliceLb =
            integerCast(loc, rewriter, idxTy, operands[sliceOffset]);
        auto adj =
            rewriter.create<mlir::LLVM::SubOp>(loc, idxTy, sliceLb, lb, nsw);
        diff = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, diff, adj, nsw);
      }
      // Update the offset given the stride and the zero based index `diff`
      // that was just computed.
      if (baseIsBoxed) {
        // Use stride in bytes from the descriptor.
        mlir::Value stride =
            getStrideFromBox(loc, baseBoxTyPair, operands[0], i, rewriter);
        auto sc =
            rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, stride, nsw);
        offset =
            rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, offset, nsw);
      } else {
        // Use stride computed at last iteration.
        auto sc =
            rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, diff, prevExt, nsw);
        offset =
            rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, offset, nsw);
        // Compute next stride assuming contiguity of the base array
        // (in element number).
        auto nextExt = integerCast(loc, rewriter, idxTy, operands[shapeOffset]);
        prevExt = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, prevExt,
                                                     nextExt, nsw);
      }
    }

    // Add computed offset to the base address.
    if (baseIsBoxed) {
      // Working with byte offsets. The base address is read from the fir.box.
      // and used in i8* GEP to do the pointer arithmetic.
      mlir::Type byteTy = ::getI8Type(coor.getContext());
      mlir::Value base =
          getBaseAddrFromBox(loc, baseBoxTyPair, operands[0], rewriter);
      llvm::SmallVector<mlir::LLVM::GEPArg> args{offset};
      auto addr = rewriter.create<mlir::LLVM::GEPOp>(loc, llvmPtrTy, byteTy,
                                                     base, args);
      if (coor.getSubcomponent().empty()) {
        rewriter.replaceOp(coor, addr);
        return mlir::success();
      }
      // Cast the element address from void* to the derived type so that the
      // derived type members can be addresses via a GEP using the index of
      // components.
      mlir::Type elementType =
          getLlvmObjectTypeFromBoxType(coor.getMemref().getType());
      while (auto arrayTy =
                 mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(elementType))
        elementType = arrayTy.getElementType();
      args.clear();
      args.push_back(0);
      if (!coor.getLenParams().empty()) {
        // If type parameters are present, then we don't want to use a GEPOp
        // as below, as the LLVM struct type cannot be statically defined.
        TODO(loc, "derived type with type parameters");
      }
      llvm::SmallVector<mlir::Value> indices = convertSubcomponentIndices(
          loc, elementType,
          operands.slice(coor.getSubcomponentOperandIndex(),
                         coor.getSubcomponent().size()));
      args.append(indices.begin(), indices.end());
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, llvmPtrTy,
                                                     elementType, addr, args);
      return mlir::success();
    }

    // The array was not boxed, so it must be contiguous. offset is therefore an
    // element offset and the base type is kept in the GEP unless the element
    // type size is itself dynamic.
    mlir::Type objectTy = fir::unwrapRefType(coor.getMemref().getType());
    mlir::Type eleType = fir::unwrapSequenceType(objectTy);
    mlir::Type gepObjectType = convertType(eleType);
    llvm::SmallVector<mlir::LLVM::GEPArg> args;
    if (coor.getSubcomponent().empty()) {
      // No subcomponent.
      if (!coor.getLenParams().empty()) {
        // Type parameters. Adjust element size explicitly.
        auto eleTy = fir::dyn_cast_ptrEleTy(coor.getType());
        assert(eleTy && "result must be a reference-like type");
        if (fir::characterWithDynamicLen(eleTy)) {
          assert(coor.getLenParams().size() == 1);
          auto length = integerCast(loc, rewriter, idxTy,
                                    operands[coor.getLenParamsOperandIndex()]);
          offset = rewriter.create<mlir::LLVM::MulOp>(loc, idxTy, offset,
                                                      length, nsw);
        } else {
          TODO(loc, "compute size of derived type with type parameters");
        }
      }
      args.push_back(offset);
    } else {
      // There are subcomponents.
      args.push_back(offset);
      llvm::SmallVector<mlir::Value> indices = convertSubcomponentIndices(
          loc, gepObjectType,
          operands.slice(coor.getSubcomponentOperandIndex(),
                         coor.getSubcomponent().size()));
      args.append(indices.begin(), indices.end());
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        coor, llvmPtrTy, gepObjectType, adaptor.getMemref(), args);
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
    : public fir::FIROpAndTypeConversion<fir::CoordinateOp> {
  using FIROpAndTypeConversion::FIROpAndTypeConversion;

  llvm::LogicalResult
  doRewrite(fir::CoordinateOp coor, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ValueRange operands = adaptor.getOperands();

    mlir::Location loc = coor.getLoc();
    mlir::Value base = operands[0];
    mlir::Type baseObjectTy = coor.getBaseType();
    mlir::Type objectTy = fir::dyn_cast_ptrOrBoxEleTy(baseObjectTy);
    assert(objectTy && "fir.coordinate_of expects a reference type");
    mlir::Type llvmObjectTy = convertType(objectTy);

    // Complex type - basically, extract the real or imaginary part
    // FIXME: double check why this is done before the fir.box case below.
    if (fir::isa_complex(objectTy)) {
      mlir::Value gep =
          genGEP(loc, llvmObjectTy, rewriter, base, 0, operands[1]);
      rewriter.replaceOp(coor, gep);
      return mlir::success();
    }

    // Boxed type - get the base pointer from the box
    if (mlir::dyn_cast<fir::BaseBoxType>(baseObjectTy))
      return doRewriteBox(coor, operands, loc, rewriter);

    // Reference, pointer or a heap type
    if (mlir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType>(
            baseObjectTy))
      return doRewriteRefOrPtr(coor, llvmObjectTy, operands, loc, rewriter);

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
    return mlir::isa<fir::SequenceType, fir::RecordType, mlir::TupleType>(type);
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
      if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(type)) {
        subEle = true;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto recTy = mlir::dyn_cast<fir::RecordType>(type)) {
        subEle = true;
        type = recTy.getType(getFieldNumber(recTy, nxtOpnd));
      } else if (auto tupTy = mlir::dyn_cast<mlir::TupleType>(type)) {
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
      if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(type)) {
        if (fir::sequenceWithNonConstantShape(arrTy))
          return false;
        i += arrTy.getDimension() - 1;
        type = arrTy.getEleTy();
      } else if (auto strTy = mlir::dyn_cast<fir::RecordType>(type)) {
        type = strTy.getType(getFieldNumber(strTy, nxtOpnd));
      } else if (auto strTy = mlir::dyn_cast<mlir::TupleType>(type)) {
        type = strTy.getType(getConstantIntValue(nxtOpnd));
      } else {
        return true;
      }
    }
    return true;
  }

private:
  llvm::LogicalResult
  doRewriteBox(fir::CoordinateOp coor, mlir::ValueRange operands,
               mlir::Location loc,
               mlir::ConversionPatternRewriter &rewriter) const {
    mlir::Type boxObjTy = coor.getBaseType();
    assert(mlir::dyn_cast<fir::BaseBoxType>(boxObjTy) &&
           "This is not a `fir.box`");
    TypePair boxTyPair = getBoxTypePair(boxObjTy);

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
        getBaseAddrFromBox(loc, boxTyPair, boxBaseAddr, rewriter);
    // Component Type
    auto cpnTy = fir::dyn_cast_ptrOrBoxEleTy(boxObjTy);
    mlir::Type llvmPtrTy = ::getLlvmPtrType(coor.getContext());
    mlir::Type byteTy = ::getI8Type(coor.getContext());
    mlir::LLVM::IntegerOverflowFlags nsw =
        mlir::LLVM::IntegerOverflowFlags::nsw;

    for (unsigned i = 1, last = operands.size(); i < last; ++i) {
      if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(cpnTy)) {
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
          mlir::Value stride = getStrideFromBox(loc, boxTyPair, operands[0],
                                                index - i, rewriter);
          auto sc = rewriter.create<mlir::LLVM::MulOp>(
              loc, idxTy, operands[index], stride, nsw);
          off = rewriter.create<mlir::LLVM::AddOp>(loc, idxTy, sc, off, nsw);
        }
        resultAddr = rewriter.create<mlir::LLVM::GEPOp>(
            loc, llvmPtrTy, byteTy, resultAddr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{off});
        i += arrTy.getDimension() - 1;
        cpnTy = arrTy.getEleTy();
      } else if (auto recTy = mlir::dyn_cast<fir::RecordType>(cpnTy)) {
        mlir::Value nxtOpnd = operands[i];
        cpnTy = recTy.getType(getFieldNumber(recTy, nxtOpnd));
        auto llvmRecTy = lowerTy().convertType(recTy);
        resultAddr = rewriter.create<mlir::LLVM::GEPOp>(
            loc, llvmPtrTy, llvmRecTy, resultAddr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{0, nxtOpnd});
      } else {
        fir::emitFatalError(loc, "unexpected type in coordinate_of");
      }
    }

    rewriter.replaceOp(coor, resultAddr);
    return mlir::success();
  }

  llvm::LogicalResult
  doRewriteRefOrPtr(fir::CoordinateOp coor, mlir::Type llvmObjectTy,
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
    if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(cpnTy)) {
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
          cpnTy = mlir::cast<fir::SequenceType>(cpnTy).getElementType();
          // append array range in reverse (FIR arrays are column-major)
          offs.append(arrIdx.rbegin(), arrIdx.rend());
          arrIdx.clear();
          dims.reset();
          continue;
        }
        if (auto arrTy = mlir::dyn_cast<fir::SequenceType>(cpnTy)) {
          int d = arrTy.getDimension() - 1;
          if (d > 0) {
            dims = d;
            arrIdx.push_back(nxtOpnd);
            continue;
          }
          cpnTy = mlir::cast<fir::SequenceType>(cpnTy).getElementType();
          offs.push_back(nxtOpnd);
          continue;
        }

        // check if the i-th coordinate relates to a field
        if (auto recTy = mlir::dyn_cast<fir::RecordType>(cpnTy))
          cpnTy = recTy.getType(getFieldNumber(recTy, nxtOpnd));
        else if (auto tupTy = mlir::dyn_cast<mlir::TupleType>(cpnTy))
          cpnTy = tupTy.getType(getConstantIntValue(nxtOpnd));
        else
          cpnTy = nullptr;

        offs.push_back(nxtOpnd);
      }
      if (dims)
        offs.append(arrIdx.rbegin(), arrIdx.rend());
      mlir::Value base = operands[0];
      mlir::Value retval = genGEP(loc, llvmObjectTy, rewriter, base, offs);
      rewriter.replaceOp(coor, retval);
      return mlir::success();
    }

    return mlir::emitError(
        loc, "fir.coordinate_of base operand has unsupported type");
  }
};

/// Convert `fir.field_index`. The conversion depends on whether the size of
/// the record is static or dynamic.
struct FieldIndexOpConversion : public fir::FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  // NB: most field references should be resolved by this point
  llvm::LogicalResult
  matchAndRewrite(fir::FieldIndexOp field, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto recTy = mlir::cast<fir::RecordType>(field.getOnType());
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
        addLLVMOpBundleAttrs(rewriter, {callAttr, fieldAttr},
                             adaptor.getOperands().size()));
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
struct FirEndOpConversion : public fir::FIROpConversion<fir::FirEndOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::FirEndOp firEnd, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(firEnd.getLoc(), "fir.end codegen");
    return mlir::failure();
  }
};

/// Lower `fir.type_desc` to a global addr.
struct TypeDescOpConversion : public fir::FIROpConversion<fir::TypeDescOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::TypeDescOp typeDescOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type inTy = typeDescOp.getInType();
    assert(mlir::isa<fir::RecordType>(inTy) && "expecting fir.type");
    auto recordType = mlir::dyn_cast<fir::RecordType>(inTy);
    auto module = typeDescOp.getOperation()->getParentOfType<mlir::ModuleOp>();
    std::string typeDescName =
        this->options.typeDescriptorsRenamedForAssembly
            ? fir::NameUniquer::getTypeDescriptorAssemblyName(
                  recordType.getName())
            : fir::NameUniquer::getTypeDescriptorName(recordType.getName());
    auto llvmPtrTy = ::getLlvmPtrType(typeDescOp.getContext());
    if (auto global = module.lookupSymbol<mlir::LLVM::GlobalOp>(typeDescName)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
          typeDescOp, llvmPtrTy, global.getSymName());
      return mlir::success();
    } else if (auto global = module.lookupSymbol<fir::GlobalOp>(typeDescName)) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
          typeDescOp, llvmPtrTy, global.getSymName());
      return mlir::success();
    }
    return mlir::failure();
  }
};

/// Lower `fir.has_value` operation to `llvm.return` operation.
struct HasValueOpConversion
    : public mlir::OpConversionPattern<fir::HasValueOp> {
  using OpConversionPattern::OpConversionPattern;

  llvm::LogicalResult
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
struct GlobalOpConversion : public fir::FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::GlobalOp global, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<mlir::Attribute> dbgExprs;

    if (auto fusedLoc = mlir::dyn_cast<mlir::FusedLoc>(global.getLoc())) {
      if (auto gvExprAttr = mlir::dyn_cast_if_present<mlir::ArrayAttr>(
              fusedLoc.getMetadata())) {
        for (auto attr : gvExprAttr.getAsRange<mlir::Attribute>())
          if (auto dbgAttr =
                  mlir::dyn_cast<mlir::LLVM::DIGlobalVariableExpressionAttr>(
                      attr))
            dbgExprs.push_back(dbgAttr);
      }
    }

    auto tyAttr = convertType(global.getType());
    if (auto boxType = mlir::dyn_cast<fir::BaseBoxType>(global.getType()))
      tyAttr = this->lowerTy().convertBoxTypeAsStruct(boxType);
    auto loc = global.getLoc();
    mlir::Attribute initAttr = global.getInitVal().value_or(mlir::Attribute());
    assert(attributeTypeIsCompatible(global.getContext(), initAttr, tyAttr));
    auto linkage = convertLinkage(global.getLinkName());
    auto isConst = global.getConstant().has_value();
    mlir::SymbolRefAttr comdat;
    llvm::ArrayRef<mlir::NamedAttribute> attrs;
    auto g = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, tyAttr, isConst, linkage, global.getSymName(), initAttr, 0, 0,
        false, false, comdat, attrs, dbgExprs);

    if (global.getAlignment() && *global.getAlignment() > 0)
      g.setAlignment(*global.getAlignment());

    auto module = global->getParentOfType<mlir::ModuleOp>();
    auto gpuMod = global->getParentOfType<mlir::gpu::GPUModuleOp>();
    // Add comdat if necessary
    if (fir::getTargetTriple(module).supportsCOMDAT() &&
        (linkage == mlir::LLVM::Linkage::Linkonce ||
         linkage == mlir::LLVM::Linkage::LinkonceODR) &&
        !gpuMod) {
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
              mlir::cast<mlir::ShapedType>(vecType), constant.getValue());
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
                        mlir::ModuleOp module) {
    const char *comdatName = "__llvm_comdat";
    mlir::LLVM::ComdatOp comdatOp =
        module.lookupSymbol<mlir::LLVM::ComdatOp>(comdatName);
    if (!comdatOp) {
      comdatOp =
          rewriter.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
    }
    if (auto select = comdatOp.lookupSymbol<mlir::LLVM::ComdatSelectorOp>(
            global.getSymName()))
      return;
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
struct LoadOpConversion : public fir::FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::LoadOp load, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Type llvmLoadTy = convertObjectType(load.getType());
    if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(load.getType())) {
      // fir.box is a special case because it is considered an ssa value in
      // fir, but it is lowered as a pointer to a descriptor. So
      // fir.ref<fir.box> and fir.box end up being the same llvm types and
      // loading a fir.ref<fir.box> is implemented as taking a snapshot of the
      // descriptor value into a new descriptor temp.
      auto inputBoxStorage = adaptor.getOperands()[0];
      mlir::Value newBoxStorage;
      mlir::Location loc = load.getLoc();
      if (auto callOp = mlir::dyn_cast_or_null<mlir::LLVM::CallOp>(
              inputBoxStorage.getDefiningOp())) {
        if (callOp.getCallee() &&
            (*callOp.getCallee())
                .starts_with(RTNAME_STRING(CUFAllocDescriptor))) {
          // CUDA Fortran local descriptor are allocated in managed memory. So
          // new storage must be allocated the same way.
          auto mod = load->getParentOfType<mlir::ModuleOp>();
          newBoxStorage =
              genCUFAllocDescriptor(loc, rewriter, mod, boxTy, lowerTy());
        }
      }
      if (!newBoxStorage)
        newBoxStorage = genAllocaAndAddrCastWithType(loc, llvmLoadTy,
                                                     defaultAlign, rewriter);

      TypePair boxTypePair{boxTy, llvmLoadTy};
      mlir::Value boxSize =
          computeBoxSize(loc, boxTypePair, inputBoxStorage, rewriter);
      auto memcpy = rewriter.create<mlir::LLVM::MemcpyOp>(
          loc, newBoxStorage, inputBoxStorage, boxSize, /*isVolatile=*/false);

      if (std::optional<mlir::ArrayAttr> optionalTag = load.getTbaa())
        memcpy.setTBAATags(*optionalTag);
      else
        attachTBAATag(memcpy, boxTy, boxTy, nullptr);
      rewriter.replaceOp(load, newBoxStorage);
    } else {
      auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(
          load.getLoc(), llvmLoadTy, adaptor.getOperands(), load->getAttrs());
      if (std::optional<mlir::ArrayAttr> optionalTag = load.getTbaa())
        loadOp.setTBAATags(*optionalTag);
      else
        attachTBAATag(loadOp, load.getType(), load.getType(), nullptr);
      rewriter.replaceOp(load, loadOp.getResult());
    }
    return mlir::success();
  }
};

/// Lower `fir.no_reassoc` to LLVM IR dialect.
/// TODO: how do we want to enforce this in LLVM-IR? Can we manipulate the fast
/// math flags?
struct NoReassocOpConversion : public fir::FIROpConversion<fir::NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
struct SelectCaseOpConversion : public fir::FIROpConversion<fir::SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectCaseOp caseOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    unsigned conds = caseOp.getNumConditions();
    llvm::ArrayRef<mlir::Attribute> cases = caseOp.getCases().getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    auto ty = caseOp.getSelector().getType();
    if (mlir::isa<fir::CharacterType>(ty)) {
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
      mlir::Attribute attr = cases[t];
      assert(mlir::isa<mlir::UnitAttr>(attr) || cmpOps.has_value());
      if (mlir::isa<fir::PointIntervalAttr>(attr)) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, selector, cmpOps->front());
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (mlir::isa<fir::LowerBoundAttr>(attr)) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, cmpOps->front(), selector);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (mlir::isa<fir::UpperBoundAttr>(attr)) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, cmpOps->front());
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (mlir::isa<fir::ClosedIntervalAttr>(attr)) {
        mlir::Value caseArg0 = *cmpOps->begin();
        auto cmp0 = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, caseArg0, selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = createBlock(rewriter, dest);
        auto *newBlock2 = createBlock(rewriter, dest);
        rewriter.setInsertionPointToEnd(thisBlock);
        rewriter.create<mlir::LLVM::CondBrOp>(loc, cmp0, newBlock1, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock1);
        mlir::Value caseArg1 = *(cmpOps->begin() + 1);
        auto cmp1 = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, caseArg1);
        genCondBrOp(loc, cmp1, dest, destOps, rewriter, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(mlir::isa<mlir::UnitAttr>(attr));
      assert((t + 1 == conds) && "unit must be last");
      genBrOp(caseOp, dest, destOps, rewriter);
    }
    return mlir::success();
  }
};

/// Helper function for converting select ops. This function converts the
/// signature of the given block. If the new block signature is different from
/// `expectedTypes`, returns "failure".
static llvm::FailureOr<mlir::Block *>
getConvertedBlock(mlir::ConversionPatternRewriter &rewriter,
                  const mlir::TypeConverter *converter,
                  mlir::Operation *branchOp, mlir::Block *block,
                  mlir::TypeRange expectedTypes) {
  assert(converter && "expected non-null type converter");
  assert(!block->isEntryBlock() && "entry blocks have no predecessors");

  // There is nothing to do if the types already match.
  if (block->getArgumentTypes() == expectedTypes)
    return block;

  // Compute the new block argument types and convert the block.
  std::optional<mlir::TypeConverter::SignatureConversion> conversion =
      converter->convertBlockSignature(block);
  if (!conversion)
    return rewriter.notifyMatchFailure(branchOp,
                                       "could not compute block signature");
  if (expectedTypes != conversion->getConvertedTypes())
    return rewriter.notifyMatchFailure(
        branchOp,
        "mismatch between adaptor operand types and computed block signature");
  return rewriter.applySignatureConversion(block, *conversion, converter);
}

template <typename OP>
static llvm::LogicalResult
selectMatchAndRewrite(const fir::LLVMTypeConverter &lowering, OP select,
                      typename OP::Adaptor adaptor,
                      mlir::ConversionPatternRewriter &rewriter,
                      const mlir::TypeConverter *converter) {
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
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      destinationsOperands.push_back(destOps ? *destOps : mlir::ValueRange{});
      auto convertedBlock =
          getConvertedBlock(rewriter, converter, select, dest,
                            mlir::TypeRange(destinationsOperands.back()));
      if (mlir::failed(convertedBlock))
        return mlir::failure();
      destinations.push_back(*convertedBlock);
      caseValues.push_back(intAttr.getInt());
      continue;
    }
    assert(mlir::dyn_cast_or_null<mlir::UnitAttr>(attr));
    assert((t + 1 == conds) && "unit must be last");
    defaultOperands = destOps ? *destOps : mlir::ValueRange{};
    auto convertedBlock = getConvertedBlock(rewriter, converter, select, dest,
                                            mlir::TypeRange(defaultOperands));
    if (mlir::failed(convertedBlock))
      return mlir::failure();
    defaultDestination = *convertedBlock;
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
  return mlir::success();
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public fir::FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return selectMatchAndRewrite<fir::SelectOp>(lowerTy(), op, adaptor,
                                                rewriter, getTypeConverter());
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public fir::FIROpConversion<fir::SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectRankOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    return selectMatchAndRewrite<fir::SelectRankOp>(
        lowerTy(), op, adaptor, rewriter, getTypeConverter());
  }
};

/// Lower `fir.select_type` to LLVM IR dialect.
struct SelectTypeOpConversion : public fir::FIROpConversion<fir::SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectTypeOp select, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::emitError(select.getLoc(),
                    "fir.select_type should have already been converted");
    return mlir::failure();
  }
};

/// `fir.store` --> `llvm.store`
struct StoreOpConversion : public fir::FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::StoreOp store, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = store.getLoc();
    mlir::Type storeTy = store.getValue().getType();
    mlir::Value llvmValue = adaptor.getValue();
    mlir::Value llvmMemref = adaptor.getMemref();
    mlir::LLVM::AliasAnalysisOpInterface newOp;
    if (auto boxTy = mlir::dyn_cast<fir::BaseBoxType>(storeTy)) {
      mlir::Type llvmBoxTy = lowerTy().convertBoxTypeAsStruct(boxTy);
      // Always use memcpy because LLVM is not as effective at optimizing
      // aggregate loads/stores as it is optimizing memcpy.
      TypePair boxTypePair{boxTy, llvmBoxTy};
      mlir::Value boxSize =
          computeBoxSize(loc, boxTypePair, llvmValue, rewriter);
      newOp = rewriter.create<mlir::LLVM::MemcpyOp>(
          loc, llvmMemref, llvmValue, boxSize, /*isVolatile=*/false);
    } else {
      newOp = rewriter.create<mlir::LLVM::StoreOp>(loc, llvmValue, llvmMemref);
    }
    if (std::optional<mlir::ArrayAttr> optionalTag = store.getTbaa())
      newOp.setTBAATags(*optionalTag);
    else
      attachTBAATag(newOp, storeTy, storeTy, nullptr);
    rewriter.eraseOp(store);
    return mlir::success();
  }
};

namespace {

/// Convert `fir.unboxchar` into two `llvm.extractvalue` instructions. One for
/// the character buffer and one for the buffer length.
struct UnboxCharOpConversion : public fir::FIROpConversion<fir::UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
struct UnboxProcOpConversion : public fir::FIROpConversion<fir::UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UnboxProcOp unboxproc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TODO(unboxproc.getLoc(), "fir.unboxproc codegen");
    return mlir::failure();
  }
};

/// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public fir::FIROpConversion<fir::UndefOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UndefOp undef, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(
        undef, convertType(undef.getType()));
    return mlir::success();
  }
};

struct ZeroOpConversion : public fir::FIROpConversion<fir::ZeroOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::ZeroOp zero, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = convertType(zero.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(zero, ty);
    return mlir::success();
  }
};

/// `fir.unreachable` --> `llvm.unreachable`
struct UnreachableOpConversion
    : public fir::FIROpConversion<fir::UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
struct IsPresentOpConversion : public fir::FIROpConversion<fir::IsPresentOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::IsPresentOp isPresent, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type idxTy = lowerTy().indexType();
    mlir::Location loc = isPresent.getLoc();
    auto ptr = adaptor.getOperands()[0];

    if (mlir::isa<fir::BoxCharType>(isPresent.getVal().getType())) {
      [[maybe_unused]] auto structTy =
          mlir::cast<mlir::LLVM::LLVMStructType>(ptr.getType());
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
/// `fir.absent !fir.ref<i64>` -->  `llvm.mlir.zero : !llvm.ptr<i64>`
struct AbsentOpConversion : public fir::FIROpConversion<fir::AbsentOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AbsentOp absent, OpAdaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type ty = convertType(absent.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(absent, ty);
    return mlir::success();
  }
};

//
// Primitive operations on Complex types
//

template <typename OPTY>
static inline mlir::LLVM::FastmathFlagsAttr getLLVMFMFAttr(OPTY op) {
  return mlir::LLVM::FastmathFlagsAttr::get(
      op.getContext(),
      mlir::arith::convertArithFastMathFlagsToLLVM(op.getFastmath()));
}

/// Generate inline code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
static mlir::LLVM::InsertValueOp
complexSum(OPTY sumop, mlir::ValueRange opnds,
           mlir::ConversionPatternRewriter &rewriter,
           const fir::LLVMTypeConverter &lowering) {
  mlir::LLVM::FastmathFlagsAttr fmf = getLLVMFMFAttr(sumop);
  mlir::Value a = opnds[0];
  mlir::Value b = opnds[1];
  auto loc = sumop.getLoc();
  mlir::Type eleTy = lowering.convertType(getComplexEleTy(sumop.getType()));
  mlir::Type ty = lowering.convertType(sumop.getType());
  auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 0);
  auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 1);
  auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 0);
  auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 1);
  auto rx = rewriter.create<LLVMOP>(loc, eleTy, x0, x1, fmf);
  auto ry = rewriter.create<LLVMOP>(loc, eleTy, y0, y1, fmf);
  auto r0 = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
  auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, r0, rx, 0);
  return rewriter.create<mlir::LLVM::InsertValueOp>(loc, r1, ry, 1);
}
} // namespace

namespace {
struct AddcOpConversion : public fir::FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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

struct SubcOpConversion : public fir::FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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
struct MulcOpConversion : public fir::FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::MulcOp mulc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: Can we use a call to __muldc3 ?
    // given: (x + iy) * (x' + iy')
    // result: (xx'-yy')+i(xy'+yx')
    mlir::LLVM::FastmathFlagsAttr fmf = getLLVMFMFAttr(mulc);
    mlir::Value a = adaptor.getOperands()[0];
    mlir::Value b = adaptor.getOperands()[1];
    auto loc = mulc.getLoc();
    mlir::Type eleTy = convertType(getComplexEleTy(mulc.getType()));
    mlir::Type ty = convertType(mulc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1, fmf);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1, fmf);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1, fmf);
    auto ri = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xy, yx, fmf);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1, fmf);
    auto rr = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, xx, yy, fmf);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ra, rr, 0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, r1, ri, 1);
    rewriter.replaceOp(mulc, r0.getResult());
    return mlir::success();
  }
};

/// Inlined complex division
struct DivcOpConversion : public fir::FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::DivcOp divc, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // TODO: Can we use a call to __divdc3 instead?
    // Just generate inline code for now.
    // given: (x + iy) / (x' + iy')
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    mlir::LLVM::FastmathFlagsAttr fmf = getLLVMFMFAttr(divc);
    mlir::Value a = adaptor.getOperands()[0];
    mlir::Value b = adaptor.getOperands()[1];
    auto loc = divc.getLoc();
    mlir::Type eleTy = convertType(getComplexEleTy(divc.getType()));
    mlir::Type ty = convertType(divc.getType());
    auto x0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 0);
    auto y0 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, a, 1);
    auto x1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 0);
    auto y1 = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, b, 1);
    auto xx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, x1, fmf);
    auto x1x1 = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x1, x1, fmf);
    auto yx = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, x1, fmf);
    auto xy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, x0, y1, fmf);
    auto yy = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y0, y1, fmf);
    auto y1y1 = rewriter.create<mlir::LLVM::FMulOp>(loc, eleTy, y1, y1, fmf);
    auto d = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, x1x1, y1y1, fmf);
    auto rrn = rewriter.create<mlir::LLVM::FAddOp>(loc, eleTy, xx, yy, fmf);
    auto rin = rewriter.create<mlir::LLVM::FSubOp>(loc, eleTy, yx, xy, fmf);
    auto rr = rewriter.create<mlir::LLVM::FDivOp>(loc, eleTy, rrn, d, fmf);
    auto ri = rewriter.create<mlir::LLVM::FDivOp>(loc, eleTy, rin, d, fmf);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r1 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ra, rr, 0);
    auto r0 = rewriter.create<mlir::LLVM::InsertValueOp>(loc, r1, ri, 1);
    rewriter.replaceOp(divc, r0.getResult());
    return mlir::success();
  }
};

/// Inlined complex negation
struct NegcOpConversion : public fir::FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
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

struct BoxOffsetOpConversion : public fir::FIROpConversion<fir::BoxOffsetOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxOffsetOp boxOffset, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Type pty = ::getLlvmPtrType(boxOffset.getContext());
    mlir::Type boxType = fir::unwrapRefType(boxOffset.getBoxRef().getType());
    mlir::Type llvmBoxTy =
        lowerTy().convertBoxTypeAsStruct(mlir::cast<fir::BaseBoxType>(boxType));
    int fieldId = boxOffset.getField() == fir::BoxFieldAttr::derived_type
                      ? getTypeDescFieldId(boxType)
                      : kAddrPosInBox;
    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(
        boxOffset, pty, llvmBoxTy, adaptor.getBoxRef(),
        llvm::ArrayRef<mlir::LLVM::GEPArg>{0, fieldId});
    return mlir::success();
  }
};

/// Conversion pattern for operation that must be dead. The information in these
/// operations is used by other operation. At this point they should not have
/// anymore uses.
/// These operations are normally dead after the pre-codegen pass.
template <typename FromOp>
struct MustBeDeadConversion : public fir::FIROpConversion<FromOp> {
  explicit MustBeDeadConversion(const fir::LLVMTypeConverter &lowering,
                                const fir::FIRToLLVMPassOptions &options)
      : fir::FIROpConversion<FromOp>(lowering, options) {}
  using OpAdaptor = typename FromOp::Adaptor;

  llvm::LogicalResult
  matchAndRewrite(FromOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (!op->getUses().empty())
      return rewriter.notifyMatchFailure(op, "op must be dead");
    rewriter.eraseOp(op);
    return mlir::success();
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

  llvm::LogicalResult
  matchAndRewrite(mlir::LLVM::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    auto callee = op.getCallee();
    if (callee)
      if (*callee == "hypotf")
        op.setCalleeAttr(mlir::SymbolRefAttr::get(op.getContext(), "_hypotf"));

    rewriter.finalizeOpModification(op);
    return mlir::success();
  }
};

class RenameMSVCLibmFuncs
    : public mlir::OpRewritePattern<mlir::LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(mlir::LLVM::LLVMFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    if (op.getSymName() == "hypotf")
      op.setSymNameAttr(rewriter.getStringAttr("_hypotf"));
    rewriter.finalizeOpModification(op);
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

    if (!forcedDataLayout.empty()) {
      llvm::DataLayout dl(forcedDataLayout);
      fir::support::setMLIRDataLayout(mod, dl);
    }

    if (!forcedTargetCPU.empty())
      fir::setTargetCPU(mod, forcedTargetCPU);

    if (!forcedTuneCPU.empty())
      fir::setTuneCPU(mod, forcedTuneCPU);

    if (!forcedTargetFeatures.empty())
      fir::setTargetFeatures(mod, forcedTargetFeatures);

    if (typeDescriptorsRenamedForAssembly)
      options.typeDescriptorsRenamedForAssembly =
          typeDescriptorsRenamedForAssembly;

    // Run dynamic pass pipeline for converting Math dialect
    // operations into other dialects (llvm, func, etc.).
    // Some conversions of Math operations cannot be done
    // by just using conversion patterns. This is true for
    // conversions that affect the ModuleOp, e.g. create new
    // function operations in it. We have to run such conversions
    // as passes here.
    mlir::OpPassManager mathConvertionPM("builtin.module");

    bool isAMDGCN = fir::getTargetTriple(mod).isAMDGCN();
    // If compiling for AMD target some math operations must be lowered to AMD
    // GPU library calls, the rest can be converted to LLVM intrinsics, which
    // is handled in the mathToLLVM conversion. The lowering to libm calls is
    // not needed since all math operations are handled this way.
    if (isAMDGCN)
      mathConvertionPM.addPass(mlir::createConvertMathToROCDL());

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

    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetMLIRDataLayout(mod, /*allowDefaultLayout=*/true);
    if (!dl) {
      mlir::emitError(mod.getLoc(),
                      "module operation must carry a data layout attribute "
                      "to generate llvm IR from FIR");
      signalPassFailure();
      return;
    }

    auto *context = getModule().getContext();
    fir::LLVMTypeConverter typeConverter{getModule(),
                                         options.applyTBAA || applyTBAA,
                                         options.forceUnifiedTBAATree, *dl};
    mlir::RewritePatternSet pattern(context);
    fir::populateFIRToLLVMConversionPatterns(typeConverter, pattern, options);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, pattern);
    mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, pattern);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, pattern);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          pattern);
    mlir::cf::populateAssertToLLVMConversionPattern(typeConverter, pattern);
    // Math operations that have not been converted yet must be converted
    // to Libm.
    if (!isAMDGCN)
      mlir::populateMathToLibmConversionPatterns(pattern);
    mlir::populateComplexToLLVMConversionPatterns(typeConverter, pattern);
    mlir::populateVectorToLLVMConversionPatterns(typeConverter, pattern);

    // Flang specific overloads for OpenMP operations, to allow for special
    // handling of things like Box types.
    fir::populateOpenMPFIRToLLVMConversionPatterns(typeConverter, pattern);

    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    // The OpenMP dialect is legal for Operations without regions, for those
    // which contains regions it is legal if the region contains only the
    // LLVM dialect. Add OpenMP dialect as a legal dialect for conversion and
    // legalize conversion of OpenMP operations without regions.
    mlir::configureOpenMPToLLVMConversionLegality(target, typeConverter);
    target.addLegalDialect<mlir::omp::OpenMPDialect>();
    target.addLegalDialect<mlir::acc::OpenACCDialect>();
    target.addLegalDialect<mlir::gpu::GPUDialect>();

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
            return *callee != "hypotf";
          });
      target.addDynamicallyLegalOp<mlir::LLVM::LLVMFuncOp>(
          [](mlir::LLVM::LLVMFuncOp op) {
            return op.getSymName() != "hypotf";
          });
    }

    // apply the patterns
    if (mlir::failed(mlir::applyFullConversion(getModule(), target,
                                               std::move(pattern)))) {
      signalPassFailure();
    }

    // Run pass to add comdats to functions that have weak linkage on relevant
    // platforms
    if (fir::getTargetTriple(mod).supportsCOMDAT()) {
      mlir::OpPassManager comdatPM("builtin.module");
      comdatPM.addPass(mlir::LLVM::createLLVMAddComdats());
      if (mlir::failed(runPipeline(comdatPM, mod)))
        return signalPassFailure();
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

void fir::populateFIRToLLVMConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    fir::FIRToLLVMPassOptions &options) {
  patterns.insert<
      AbsentOpConversion, AddcOpConversion, AddrOfOpConversion,
      AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
      BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
      BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
      BoxOffsetOpConversion, BoxProcHostOpConversion, BoxRankOpConversion,
      BoxTypeCodeOpConversion, BoxTypeDescOpConversion, CallOpConversion,
      CmpcOpConversion, ConvertOpConversion, CoordinateOpConversion,
      DTEntryOpConversion, DeclareOpConversion, DivcOpConversion,
      EmboxOpConversion, EmboxCharOpConversion, EmboxProcOpConversion,
      ExtractValueOpConversion, FieldIndexOpConversion, FirEndOpConversion,
      FreeMemOpConversion, GlobalLenOpConversion, GlobalOpConversion,
      InsertOnRangeOpConversion, IsPresentOpConversion,
      LenParamIndexOpConversion, LoadOpConversion, MulcOpConversion,
      NegcOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
      SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
      ShapeOpConversion, ShapeShiftOpConversion, ShiftOpConversion,
      SliceOpConversion, StoreOpConversion, StringLitOpConversion,
      SubcOpConversion, TypeDescOpConversion, TypeInfoOpConversion,
      UnboxCharOpConversion, UnboxProcOpConversion, UndefOpConversion,
      UnreachableOpConversion, XArrayCoorOpConversion, XEmboxOpConversion,
      XReboxOpConversion, ZeroOpConversion>(converter, options);

  // Patterns that are populated without a type converter do not trigger
  // target materializations for the operands of the root op.
  patterns.insert<HasValueOpConversion, InsertValueOpConversion>(
      patterns.getContext());
}
