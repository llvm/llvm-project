//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/CodeGen/CodeGenOpenMP.h"
#include "flang/Optimizer/CodeGen/FIROpPatterns.h"
#include "flang/Optimizer/CodeGen/LLVMInsertChainFolder.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRCG/CGOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
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
#include "aiir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "aiir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "aiir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "aiir/Conversion/ComplexToROCDLLibraryCalls/ComplexToROCDLLibraryCalls.h"
#include "aiir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "aiir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "aiir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "aiir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "aiir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Conversion/MathToFuncs/MathToFuncs.h"
#include "aiir/Conversion/MathToLLVM/MathToLLVM.h"
#include "aiir/Conversion/MathToLibm/MathToLibm.h"
#include "aiir/Conversion/MathToNVVM/MathToNVVM.h"
#include "aiir/Conversion/MathToROCDL/MathToROCDL.h"
#include "aiir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "aiir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/NVVMDialect.h"
#include "aiir/Dialect/LLVMIR/Transforms/AddComdats.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"
#include "aiir/Dialect/OpenMP/OpenMPDialect.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Matchers.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Pass/PassManager.h"
#include "aiir/Target/LLVMIR/Import.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"
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

static inline aiir::Type getLlvmPtrType(aiir::AIIRContext *context,
                                        unsigned addressSpace = 0) {
  return aiir::LLVM::LLVMPointerType::get(context, addressSpace);
}

static inline aiir::Type getI8Type(aiir::AIIRContext *context) {
  return aiir::IntegerType::get(context, 8);
}

static aiir::Block *createBlock(aiir::ConversionPatternRewriter &rewriter,
                                aiir::Block *insertBefore) {
  assert(insertBefore && "expected valid insertion block");
  return rewriter.createBlock(insertBefore->getParent(),
                              aiir::Region::iterator(insertBefore));
}

/// Extract constant from a value that must be the result of one of the
/// ConstantOp operations.
static int64_t getConstantIntValue(aiir::Value val) {
  if (auto constVal = fir::getIntIfConstant(val))
    return *constVal;
  fir::emitFatalError(val.getLoc(), "must be a constant");
}

static unsigned getTypeDescFieldId(aiir::Type ty) {
  auto isArray = aiir::isa<fir::SequenceType>(fir::dyn_cast_ptrOrBoxEleTy(ty));
  return isArray ? kOptTypePtrPosInBox : kDimsPosInBox;
}
static unsigned getLenParamFieldId(aiir::Type ty) {
  return getTypeDescFieldId(ty) + 1;
}

/// Set LLVM alignment operand attributes on a memcpy op using the ABI
/// alignment of the given type (for dst and src; size has no alignment).
static void setMemcpyAlignmentArgAttrs(
    aiir::LLVM::MemcpyOp memcpy, aiir::ConversionPatternRewriter &rewriter,
    const aiir::DataLayout &dataLayout, aiir::Type llvmType) {
  unsigned alignValue = dataLayout.getTypeABIAlignment(llvmType);
  aiir::IntegerAttr alignAttr = rewriter.getI64IntegerAttr(alignValue);
  aiir::NamedAttribute alignNamedAttr(
      aiir::StringAttr::get(rewriter.getContext(),
                            aiir::LLVM::LLVMDialect::getAlignAttrName()),
      alignAttr);
  aiir::DictionaryAttr alignDict = rewriter.getDictionaryAttr(alignNamedAttr);
  memcpy.setArgAttrsAttr(rewriter.getArrayAttr(
      {alignDict, alignDict, rewriter.getDictionaryAttr({})}));
}

static llvm::SmallVector<aiir::NamedAttribute>
addLLVMOpBundleAttrs(aiir::ConversionPatternRewriter &rewriter,
                     llvm::ArrayRef<aiir::NamedAttribute> attrs,
                     int32_t numCallOperands) {
  llvm::SmallVector<aiir::NamedAttribute> newAttrs;
  newAttrs.reserve(attrs.size() + 2);

  for (aiir::NamedAttribute attr : attrs) {
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

// Replaces an existing operation with an AddressOfOp or an AddrSpaceCastOp
// depending on the existing address spaces of the type.
aiir::Value replaceWithAddrOfOrASCast(aiir::ConversionPatternRewriter &rewriter,
                                      aiir::Location loc,
                                      std::uint64_t globalAS,
                                      std::uint64_t programAS,
                                      llvm::StringRef symName, aiir::Type type,
                                      aiir::Operation *replaceOp = nullptr) {
  if (aiir::isa<aiir::LLVM::LLVMPointerType>(type)) {
    if (globalAS != programAS) {
      auto llvmAddrOp = aiir::LLVM::AddressOfOp::create(
          rewriter, loc, getLlvmPtrType(rewriter.getContext(), globalAS),
          symName);
      if (replaceOp)
        return rewriter.replaceOpWithNewOp<aiir::LLVM::AddrSpaceCastOp>(
            replaceOp, ::getLlvmPtrType(rewriter.getContext(), programAS),
            llvmAddrOp);
      return aiir::LLVM::AddrSpaceCastOp::create(
          rewriter, loc, getLlvmPtrType(rewriter.getContext(), programAS),
          llvmAddrOp);
    }

    if (replaceOp)
      return rewriter.replaceOpWithNewOp<aiir::LLVM::AddressOfOp>(
          replaceOp, getLlvmPtrType(rewriter.getContext(), globalAS), symName);
    return aiir::LLVM::AddressOfOp::create(
        rewriter, loc, getLlvmPtrType(rewriter.getContext(), globalAS),
        symName);
  }

  if (replaceOp)
    return rewriter.replaceOpWithNewOp<aiir::LLVM::AddressOfOp>(replaceOp, type,
                                                                symName);
  return aiir::LLVM::AddressOfOp::create(rewriter, loc, type, symName);
}

/// Lower `fir.address_of` operation to `llvm.address_of` operation.
struct AddrOfOpConversion : public fir::FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AddrOfOp addr, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {

    if (auto gpuMod = addr->getParentOfType<aiir::gpu::GPUModuleOp>()) {
      auto global = gpuMod.lookupSymbol<aiir::LLVM::GlobalOp>(addr.getSymbol());
      replaceWithAddrOfOrASCast(
          rewriter, addr->getLoc(),
          global ? global.getAddrSpace() : getGlobalAddressSpace(rewriter),
          getProgramAddressSpace(rewriter),
          global ? global.getSymName()
                 : addr.getSymbol().getRootReference().getValue(),
          convertType(addr.getType()), addr);
      return aiir::success();
    }

    auto global = addr->getParentOfType<aiir::ModuleOp>()
                      .lookupSymbol<aiir::LLVM::GlobalOp>(addr.getSymbol());
    replaceWithAddrOfOrASCast(
        rewriter, addr->getLoc(),
        global ? global.getAddrSpace() : getGlobalAddressSpace(rewriter),
        getProgramAddressSpace(rewriter),
        global ? global.getSymName()
               : addr.getSymbol().getRootReference().getValue(),
        convertType(addr.getType()), addr);
    return aiir::success();
  }
};
} // namespace

/// Lookup the function to compute the memory size of this parametric derived
/// type. The size of the object may depend on the LEN type parameters of the
/// derived type.
static aiir::LLVM::LLVMFuncOp
getDependentTypeMemSizeFn(fir::RecordType recTy, fir::AllocaOp op,
                          aiir::ConversionPatternRewriter &rewriter) {
  auto module = op->getParentOfType<aiir::ModuleOp>();
  std::string name = recTy.getName().str() + "P.mem.size";
  if (auto memSizeFunc = module.lookupSymbol<aiir::LLVM::LLVMFuncOp>(name))
    return memSizeFunc;
  TODO(op.getLoc(), "did not find allocation function");
}

namespace {
struct DeclareOpConversion : public fir::FIROpConversion<fir::cg::XDeclareOp> {
public:
  using FIROpConversion::FIROpConversion;
  llvm::LogicalResult
  matchAndRewrite(fir::cg::XDeclareOp declareOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    auto memRef = adaptor.getOperands()[0];
    if (auto fusedLoc = aiir::dyn_cast<aiir::FusedLoc>(declareOp.getLoc())) {
      if (auto varAttr =
              aiir::dyn_cast_or_null<aiir::LLVM::DILocalVariableAttr>(
                  fusedLoc.getMetadata())) {
        aiir::LLVM::DbgDeclareOp::create(rewriter, memRef.getLoc(), memRef,
                                         varAttr, nullptr);
      }
    }
    rewriter.replaceOp(declareOp, memRef);
    return aiir::success();
  }
};

struct DeclareValueOpConversion
    : public fir::FIROpConversion<fir::DeclareValueOp> {
public:
  using FIROpConversion::FIROpConversion;
  llvm::LogicalResult
  matchAndRewrite(fir::DeclareValueOp declareOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    auto value = adaptor.getOperands()[0];
    if (auto fusedLoc = aiir::dyn_cast<aiir::FusedLoc>(declareOp.getLoc())) {
      if (auto varAttr =
              aiir::dyn_cast_or_null<aiir::LLVM::DILocalVariableAttr>(
                  fusedLoc.getMetadata())) {
        aiir::LLVM::DbgValueOp::create(rewriter, value.getLoc(), value, varAttr,
                                       nullptr);
      }
    }
    rewriter.eraseOp(declareOp);
    return aiir::success();
  }
};
} // namespace

namespace {
/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public fir::FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AllocaOp alloc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();
    auto loc = alloc.getLoc();
    aiir::Type ity = lowerTy().indexType();
    unsigned i = 0;
    aiir::Value size = fir::genConstantIndex(loc, ity, rewriter, 1).getResult();
    aiir::Type firObjType = fir::unwrapRefType(alloc.getType());
    aiir::Type llvmObjectType = convertObjectType(firObjType);
    if (alloc.hasLenParams()) {
      unsigned end = alloc.numLenParams();
      llvm::SmallVector<aiir::Value> lenParams;
      for (; i < end; ++i)
        lenParams.push_back(operands[i]);
      aiir::Type scalarType = fir::unwrapSequenceType(alloc.getInType());
      if (auto chrTy = aiir::dyn_cast<fir::CharacterType>(scalarType)) {
        fir::CharacterType rawCharTy = fir::CharacterType::getUnknownLen(
            chrTy.getContext(), chrTy.getFKind());
        llvmObjectType = convertType(rawCharTy);
        assert(end == 1);
        size = integerCast(loc, rewriter, ity, lenParams[0], /*fold=*/true);
      } else if (auto recTy = aiir::dyn_cast<fir::RecordType>(scalarType)) {
        aiir::LLVM::LLVMFuncOp memSizeFn =
            getDependentTypeMemSizeFn(recTy, alloc, rewriter);
        if (!memSizeFn)
          emitError(loc, "did not find allocation function");
        aiir::NamedAttribute attr = rewriter.getNamedAttr(
            "callee", aiir::SymbolRefAttr::get(memSizeFn));
        auto call = aiir::LLVM::CallOp::create(
            rewriter, loc, ity, lenParams,
            addLLVMOpBundleAttrs(rewriter, {attr}, lenParams.size()));
        size = call.getResult();
        llvmObjectType = ::getI8Type(alloc.getContext());
      } else {
        return emitError(loc, "unexpected type ")
               << scalarType << " with type parameters";
      }
    }
    if (auto scaleSize = fir::genAllocationScaleSize(
            alloc.getLoc(), alloc.getInType(), ity, rewriter))
      size =
          rewriter.createOrFold<aiir::LLVM::MulOp>(loc, ity, size, scaleSize);
    if (alloc.hasShapeOperands()) {
      unsigned end = operands.size();
      for (; i < end; ++i)
        size = rewriter.createOrFold<aiir::LLVM::MulOp>(
            loc, ity, size,
            integerCast(loc, rewriter, ity, operands[i], /*fold=*/true));
    }

    unsigned allocaAs = getAllocaAddressSpace(rewriter);
    unsigned programAs = getProgramAddressSpace(rewriter);

    // A value defined by a block arg, such as fir.if for a
    // optional assumed character dummy len, doesn't have a defining op.
    if (aiir::isa_and_nonnull<aiir::LLVM::ConstantOp>(size.getDefiningOp())) {
      // Set the Block in which the llvm alloca should be inserted.
      aiir::Operation *parentOp = rewriter.getInsertionBlock()->getParentOp();
      aiir::Region *parentRegion = rewriter.getInsertionBlock()->getParent();
      aiir::Block *insertBlock =
          getBlockForAllocaInsert(parentOp, parentRegion);

      // The old size might have had multiple users, some at a broader scope
      // than we can safely outline the alloca to. As it is only an
      // llvm.constant operation, it is faster to clone it than to calculate the
      // dominance to see if it really should be moved.
      aiir::Operation *clonedSize = rewriter.clone(*size.getDefiningOp());
      size = clonedSize->getResult(0);
      clonedSize->moveBefore(&insertBlock->front());
      rewriter.setInsertionPointAfter(size.getDefiningOp());
    }

    // NOTE: we used to pass alloc->getAttrs() in the builder for non opaque
    // pointers! Only propagate pinned and bindc_name to help debugging, but
    // this should have no functional purpose (and passing the operand segment
    // attribute like before is certainly bad).
    auto llvmAlloc = aiir::LLVM::AllocaOp::create(
        rewriter, loc, ::getLlvmPtrType(alloc.getContext(), allocaAs),
        llvmObjectType, size);
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
      rewriter.replaceOpWithNewOp<aiir::LLVM::AddrSpaceCastOp>(
          alloc, ::getLlvmPtrType(alloc.getContext(), programAs), llvmAlloc);
    }

    return aiir::success();
  }
};
} // namespace

namespace {

static bool isInGlobalOp(aiir::ConversionPatternRewriter &rewriter) {
  auto *thisBlock = rewriter.getInsertionBlock();
  return thisBlock && aiir::isa<aiir::LLVM::GlobalOp>(thisBlock->getParentOp());
}

// Inside a fir.global, the input box was produced as an llvm.struct<>
// because objects cannot be handled in memory inside a fir.global body that
// must be constant foldable. However, the type translation are not
// contextual, so the fir.box<T> type of the operation that produced the
// fir.box was translated to an llvm.ptr<llvm.struct<>> and the AIIR pass
// manager inserted a builtin.unrealized_conversion_cast that was inserted
// and needs to be removed here.
// This should be called by any pattern operating on operations that are
// accepting fir.box inputs and are used in fir.global.
static aiir::Value
fixBoxInputInsideGlobalOp(aiir::ConversionPatternRewriter &rewriter,
                          aiir::Value box) {
  if (isInGlobalOp(rewriter))
    if (auto unrealizedCast =
            box.getDefiningOp<aiir::UnrealizedConversionCastOp>())
      return unrealizedCast.getInputs()[0];
  return box;
}

/// Lower `fir.box_addr` to the sequence of operations to extract the first
/// element of the box.
struct BoxAddrOpConversion : public fir::FIROpConversion<fir::BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxAddrOp boxaddr, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value a = adaptor.getOperands()[0];
    auto loc = boxaddr.getLoc();
    if (auto argty =
            aiir::dyn_cast<fir::BaseBoxType>(boxaddr.getVal().getType())) {
      a = fixBoxInputInsideGlobalOp(rewriter, a);
      TypePair boxTyPair = getBoxTypePair(argty);
      rewriter.replaceOp(boxaddr,
                         getBaseAddrFromBox(loc, boxTyPair, a, rewriter));
    } else {
      rewriter.replaceOpWithNewOp<aiir::LLVM::ExtractValueOp>(boxaddr, a, 0);
    }
    return aiir::success();
  }
};

/// Convert `!fir.boxchar_len` to  `!llvm.extractvalue` for the 2nd part of the
/// boxchar.
struct BoxCharLenOpConversion : public fir::FIROpConversion<fir::BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxCharLenOp boxCharLen, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value boxChar = adaptor.getOperands()[0];
    aiir::Location loc = boxChar.getLoc();
    aiir::Type returnValTy = boxCharLen.getResult().getType();

    constexpr int boxcharLenIdx = 1;
    auto len = aiir::LLVM::ExtractValueOp::create(rewriter, loc, boxChar,
                                                  boxcharLenIdx);
    aiir::Value lenAfterCast = integerCast(loc, rewriter, returnValTy, len);
    rewriter.replaceOp(boxCharLen, lenAfterCast);

    return aiir::success();
  }
};

/// Lower `fir.box_dims` to a sequence of operations to extract the requested
/// dimension information from the boxed value.
/// Result in a triple set of GEPs and loads.
struct BoxDimsOpConversion : public fir::FIROpConversion<fir::BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxDimsOp boxdims, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<aiir::Type, 3> resultTypes = {
        convertType(boxdims.getResult(0).getType()),
        convertType(boxdims.getResult(1).getType()),
        convertType(boxdims.getResult(2).getType()),
    };
    TypePair boxTyPair = getBoxTypePair(boxdims.getVal().getType());
    auto results = getDimsFromBox(boxdims.getLoc(), resultTypes, boxTyPair,
                                  adaptor.getOperands()[0],
                                  adaptor.getOperands()[1], rewriter);
    rewriter.replaceOp(boxdims, results);
    return aiir::success();
  }
};

/// Lower `fir.box_elesize` to a sequence of operations ro extract the size of
/// an element in the boxed value.
struct BoxEleSizeOpConversion : public fir::FIROpConversion<fir::BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxEleSizeOp boxelesz, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value box = adaptor.getOperands()[0];
    auto loc = boxelesz.getLoc();
    auto ty = convertType(boxelesz.getType());
    TypePair boxTyPair = getBoxTypePair(boxelesz.getVal().getType());
    auto elemSize = getElementSizeFromBox(loc, ty, boxTyPair, box, rewriter);
    rewriter.replaceOp(boxelesz, elemSize);
    return aiir::success();
  }
};

/// Lower `fir.box_isalloc` to a sequence of operations to determine if the
/// boxed value was from an ALLOCATABLE entity.
struct BoxIsAllocOpConversion : public fir::FIROpConversion<fir::BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxIsAllocOp boxisalloc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value box = adaptor.getOperands()[0];
    auto loc = boxisalloc.getLoc();
    TypePair boxTyPair = getBoxTypePair(boxisalloc.getVal().getType());
    aiir::Value check =
        genBoxAttributeCheck(loc, boxTyPair, box, rewriter, kAttrAllocatable);
    rewriter.replaceOp(boxisalloc, check);
    return aiir::success();
  }
};

/// Lower `fir.box_isarray` to a sequence of operations to determine if the
/// boxed is an array.
struct BoxIsArrayOpConversion : public fir::FIROpConversion<fir::BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxIsArrayOp boxisarray, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value a = adaptor.getOperands()[0];
    auto loc = boxisarray.getLoc();
    TypePair boxTyPair = getBoxTypePair(boxisarray.getVal().getType());
    aiir::Value rank = getRankFromBox(loc, boxTyPair, a, rewriter);
    aiir::Value c0 = fir::genConstantIndex(loc, rank.getType(), rewriter, 0);
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        boxisarray, aiir::LLVM::ICmpPredicate::ne, rank, c0);
    return aiir::success();
  }
};

/// Lower `fir.box_isptr` to a sequence of operations to determined if the
/// boxed value was from a POINTER entity.
struct BoxIsPtrOpConversion : public fir::FIROpConversion<fir::BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxIsPtrOp boxisptr, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value box = adaptor.getOperands()[0];
    auto loc = boxisptr.getLoc();
    TypePair boxTyPair = getBoxTypePair(boxisptr.getVal().getType());
    aiir::Value check =
        genBoxAttributeCheck(loc, boxTyPair, box, rewriter, kAttrPointer);
    rewriter.replaceOp(boxisptr, check);
    return aiir::success();
  }
};

/// Lower `fir.box_rank` to the sequence of operation to extract the rank from
/// the box.
struct BoxRankOpConversion : public fir::FIROpConversion<fir::BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxRankOp boxrank, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value a = adaptor.getOperands()[0];
    auto loc = boxrank.getLoc();
    aiir::Type ty = convertType(boxrank.getType());
    TypePair boxTyPair =
        getBoxTypePair(fir::unwrapRefType(boxrank.getBox().getType()));
    aiir::Value rank = getRankFromBox(loc, boxTyPair, a, rewriter);
    aiir::Value result = integerCast(loc, rewriter, ty, rank);
    rewriter.replaceOp(boxrank, result);
    return aiir::success();
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    TODO(boxprochost.getLoc(), "fir.boxproc_host codegen");
    return aiir::failure();
  }
};

/// Lower `fir.box_tdesc` to the sequence of operations to extract the type
/// descriptor from the box.
struct BoxTypeDescOpConversion
    : public fir::FIROpConversion<fir::BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxTypeDescOp boxtypedesc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value box = adaptor.getOperands()[0];
    TypePair boxTyPair = getBoxTypePair(boxtypedesc.getBox().getType());
    auto typeDescAddr =
        loadTypeDescAddress(boxtypedesc.getLoc(), boxTyPair, box, rewriter);
    rewriter.replaceOp(boxtypedesc, typeDescAddr);
    return aiir::success();
  }
};

/// Lower `fir.box_typecode` to a sequence of operations to extract the type
/// code in the boxed value.
struct BoxTypeCodeOpConversion
    : public fir::FIROpConversion<fir::BoxTypeCodeOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxTypeCodeOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Value box = adaptor.getOperands()[0];
    auto loc = box.getLoc();
    auto ty = convertType(op.getType());
    TypePair boxTyPair = getBoxTypePair(op.getBox().getType());
    auto typeCode =
        getValueFromBox(loc, boxTyPair, box, ty, rewriter, kTypePosInBox);
    rewriter.replaceOp(op, typeCode);
    return aiir::success();
  }
};

/// Lower `fir.string_lit` to LLVM IR dialect operation.
struct StringLitOpConversion : public fir::FIROpConversion<fir::StringLitOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::StringLitOp constop, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    auto ty = convertType(constop.getType());
    auto attr = constop.getValue();
    if (aiir::isa<aiir::StringAttr>(attr)) {
      rewriter.replaceOpWithNewOp<aiir::LLVM::ConstantOp>(constop, ty, attr);
      return aiir::success();
    }

    auto charTy = aiir::cast<fir::CharacterType>(constop.getType());
    unsigned bits = lowerTy().characterBitsize(charTy);
    aiir::Type intTy = rewriter.getIntegerType(bits);
    aiir::Location loc = constop.getLoc();
    aiir::Value cst = aiir::LLVM::UndefOp::create(rewriter, loc, ty);
    if (auto arr = aiir::dyn_cast<aiir::DenseElementsAttr>(attr)) {
      cst = aiir::LLVM::ConstantOp::create(rewriter, loc, ty, arr);
    } else if (auto arr = aiir::dyn_cast<aiir::ArrayAttr>(attr)) {
      for (auto a : llvm::enumerate(arr.getValue())) {
        // convert each character to a precise bitsize
        auto elemAttr = aiir::IntegerAttr::get(
            intTy,
            aiir::cast<aiir::IntegerAttr>(a.value()).getValue().zextOrTrunc(
                bits));
        auto elemCst =
            aiir::LLVM::ConstantOp::create(rewriter, loc, intTy, elemAttr);
        cst = aiir::LLVM::InsertValueOp::create(rewriter, loc, cst, elemCst,
                                                a.index());
      }
    } else {
      return aiir::failure();
    }
    rewriter.replaceOp(constop, cst);
    return aiir::success();
  }
};

/// `fir.call` -> `llvm.call`
struct CallOpConversion : public fir::FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::CallOp call, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<aiir::Type> resultTys;
    aiir::Attribute memAttr =
        call->getAttr(fir::FIROpsDialect::getFirCallMemoryAttrName());
    if (memAttr)
      call->removeAttr(fir::FIROpsDialect::getFirCallMemoryAttrName());

    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    // Convert arith::FastMathFlagsAttr to LLVM::FastMathFlagsAttr.
    aiir::arith::AttrConvertFastMathToLLVM<fir::CallOp, aiir::LLVM::CallOp>
        attrConvert(call);
    auto llvmCall = rewriter.replaceOpWithNewOp<aiir::LLVM::CallOp>(
        call, resultTys, adaptor.getOperands(),
        addLLVMOpBundleAttrs(rewriter, attrConvert.getAttrs(),
                             adaptor.getOperands().size()));
    if (aiir::ArrayAttr argAttrsArray = call.getArgAttrsAttr()) {
      // sret and byval type needs to be converted.
      auto convertTypeAttr = [&](const aiir::NamedAttribute &attr) {
        return aiir::TypeAttr::get(convertType(
            llvm::cast<aiir::TypeAttr>(attr.getValue()).getValue()));
      };
      llvm::SmallVector<aiir::Attribute> newArgAttrsArray;
      for (auto argAttrs : argAttrsArray) {
        llvm::SmallVector<aiir::NamedAttribute> convertedAttrs;
        for (const aiir::NamedAttribute &attr :
             llvm::cast<aiir::DictionaryAttr>(argAttrs)) {
          if (attr.getName().getValue() ==
              aiir::LLVM::LLVMDialect::getByValAttrName()) {
            convertedAttrs.push_back(rewriter.getNamedAttr(
                aiir::LLVM::LLVMDialect::getByValAttrName(),
                convertTypeAttr(attr)));
          } else if (attr.getName().getValue() ==
                     aiir::LLVM::LLVMDialect::getStructRetAttrName()) {
            convertedAttrs.push_back(rewriter.getNamedAttr(
                aiir::LLVM::LLVMDialect::getStructRetAttrName(),
                convertTypeAttr(attr)));
          } else {
            convertedAttrs.push_back(attr);
          }
        }
        newArgAttrsArray.emplace_back(
            aiir::DictionaryAttr::get(rewriter.getContext(), convertedAttrs));
      }
      llvmCall.setArgAttrsAttr(rewriter.getArrayAttr(newArgAttrsArray));
    }
    if (aiir::ArrayAttr resAttrs = call.getResAttrsAttr())
      llvmCall.setResAttrsAttr(resAttrs);

    if (auto inlineAttr = call.getInlineAttrAttr()) {
      llvmCall->removeAttr("inline_attr");
      if (inlineAttr.getValue() == fir::FortranInlineEnum::no_inline) {
        llvmCall.setNoInlineAttr(rewriter.getUnitAttr());
      } else if (inlineAttr.getValue() == fir::FortranInlineEnum::inline_hint) {
        llvmCall.setInlineHintAttr(rewriter.getUnitAttr());
      } else if (inlineAttr.getValue() ==
                 fir::FortranInlineEnum::always_inline) {
        llvmCall.setAlwaysInlineAttr(rewriter.getUnitAttr());
      }
    }

    if (std::optional<aiir::ArrayAttr> optionalAccessGroups =
            call.getAccessGroups())
      llvmCall.setAccessGroups(*optionalAccessGroups);

    if (memAttr)
      llvmCall.setMemoryEffectsAttr(
          aiir::cast<aiir::LLVM::MemoryEffectsAttr>(memAttr));
    return aiir::success();
  }
};
} // namespace

static aiir::Type getComplexEleTy(aiir::Type complex) {
  return aiir::cast<aiir::ComplexType>(complex).getElementType();
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();
    aiir::Type resTy = convertType(cmp.getType());
    aiir::Location loc = cmp.getLoc();
    aiir::LLVM::FastmathFlags fmf =
        aiir::arith::convertArithFastMathFlagsToLLVM(cmp.getFastmath());
    aiir::LLVM::FCmpPredicate pred =
        static_cast<aiir::LLVM::FCmpPredicate>(cmp.getPredicate());
    auto rcp = aiir::LLVM::FCmpOp::create(
        rewriter, loc, resTy, pred,
        aiir::LLVM::ExtractValueOp::create(rewriter, loc, operands[0], 0),
        aiir::LLVM::ExtractValueOp::create(rewriter, loc, operands[1], 0), fmf);
    auto icp = aiir::LLVM::FCmpOp::create(
        rewriter, loc, resTy, pred,
        aiir::LLVM::ExtractValueOp::create(rewriter, loc, operands[0], 1),
        aiir::LLVM::ExtractValueOp::create(rewriter, loc, operands[1], 1), fmf);
    llvm::SmallVector<aiir::Value, 2> cp = {rcp, icp};
    switch (cmp.getPredicate()) {
    case aiir::arith::CmpFPredicate::OEQ: // .EQ.
      rewriter.replaceOpWithNewOp<aiir::LLVM::AndOp>(cmp, resTy, cp);
      break;
    case aiir::arith::CmpFPredicate::UNE: // .NE.
      rewriter.replaceOpWithNewOp<aiir::LLVM::OrOp>(cmp, resTy, cp);
      break;
    default:
      rewriter.replaceOp(cmp, rcp.getResult());
      break;
    }
    return aiir::success();
  }
};

/// fir.volatile_cast is only useful at the fir level. Once we lower to LLVM,
/// volatility is described by setting volatile attributes on the LLVM ops.
struct VolatileCastOpConversion
    : public fir::FIROpConversion<fir::VolatileCastOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::VolatileCastOp volatileCast, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(volatileCast, adaptor.getOperands()[0]);
    return aiir::success();
  }
};

/// Lower `fir.assumed_size_extent` to constant -1 of index type.
struct AssumedSizeExtentOpConversion
    : public fir::FIROpConversion<fir::AssumedSizeExtentOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AssumedSizeExtentOp op, OpAdaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = op.getLoc();
    aiir::Type ity = lowerTy().indexType();
    auto cst = fir::genConstantIndex(loc, ity, rewriter, -1);
    rewriter.replaceOp(op, cst.getResult());
    return aiir::success();
  }
};

/// Lower `fir.is_assumed_size_extent` to integer equality with -1.
struct IsAssumedSizeExtentOpConversion
    : public fir::FIROpConversion<fir::IsAssumedSizeExtentOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::IsAssumedSizeExtentOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = op.getLoc();
    aiir::Value val = adaptor.getVal();
    aiir::Type valTy = val.getType();
    // Create constant -1 of the operand type.
    auto negOneAttr = rewriter.getIntegerAttr(valTy, -1);
    auto negOne =
        aiir::LLVM::ConstantOp::create(rewriter, loc, valTy, negOneAttr);
    auto cmp = aiir::LLVM::ICmpOp::create(
        rewriter, loc, aiir::LLVM::ICmpPredicate::eq, val, negOne);
    rewriter.replaceOp(op, cmp.getResult());
    return aiir::success();
  }
};

/// Bitcast between types of the same bit size.
struct BitcastOpConversion : public fir::FIROpConversion<fir::BitcastOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BitcastOp bitcast, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    auto fromTy = convertType(bitcast.getValue().getType());
    auto toTy = convertType(bitcast.getRes().getType());
    aiir::Value op0 = adaptor.getOperands()[0];
    if (fromTy == toTy) {
      rewriter.replaceOp(bitcast, op0);
      return aiir::success();
    }
    aiir::Location loc = bitcast.getLoc();
    bool fromChar = aiir::isa<fir::CharacterType>(bitcast.getValue().getType());
    bool toChar = aiir::isa<fir::CharacterType>(bitcast.getRes().getType());
    aiir::Value cast = op0;
    aiir::Type scalarFromTy = fromTy;
    if (fromChar) {
      cast = aiir::LLVM::ExtractValueOp::create(rewriter, loc, cast, {0});
      scalarFromTy = cast.getType();
    }
    aiir::Type scalarToTy = toTy;
    if (toChar)
      scalarToTy = aiir::cast<aiir::LLVM::LLVMArrayType>(toTy).getElementType();

    if (scalarFromTy != scalarToTy)
      cast = aiir::LLVM::BitcastOp::create(rewriter, loc, scalarToTy, cast);

    if (toChar) {
      aiir::Value undef = aiir::LLVM::UndefOp::create(rewriter, loc, toTy);
      llvm::SmallVector<int64_t> position{0};
      cast = aiir::LLVM::InsertValueOp::create(rewriter, loc, undef, cast,
                                               position);
    }
    rewriter.replaceOp(bitcast, cast);
    return aiir::success();
  }
};

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public fir::FIROpConversion<fir::ConvertOp> {
  using FIROpConversion::FIROpConversion;

  static bool isFloatingPointTy(aiir::Type ty) {
    return aiir::isa<aiir::FloatType>(ty);
  }

  llvm::LogicalResult
  matchAndRewrite(fir::ConvertOp convert, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    auto fromFirTy = convert.getValue().getType();
    auto toFirTy = convert.getRes().getType();

    // Handle conversions between pointer-like values and memref descriptors.
    // These are produced by FIR-to-MemRef lowering and represent descriptor
    // conversion rather than pure value conversions.
    if (auto memRefTy = aiir::dyn_cast<aiir::MemRefType>(toFirTy)) {
      aiir::Location loc = convert.getLoc();
      aiir::Value basePtr = adaptor.getValue();
      assert(basePtr && "null base pointer");

      auto [strides, offset] = memRefTy.getStridesAndOffset();
      bool hasStaticLayout =
          aiir::ShapedType::isStatic(offset) &&
          llvm::none_of(strides, aiir::ShapedType::isDynamic);

      auto *firConv =
          static_cast<const fir::LLVMTypeConverter *>(this->getTypeConverter());
      assert(firConv && "expected non-null LLVMTypeConverter");

      if (memRefTy.hasStaticShape() && hasStaticLayout) {
        // Static shape and layout: build a fully-populated descriptor.
        aiir::Value memrefDesc = aiir::MemRefDescriptor::fromStaticShape(
            rewriter, loc, *firConv, memRefTy, basePtr);
        rewriter.replaceOp(convert, memrefDesc);
        return aiir::success();
      }

      // Dynamic shape or layout: create an LLVM memref descriptor and insert
      // the base pointer field, letting the rest of the fields be populated
      // by subsequent lowering.
      aiir::Type llvmMemRefTy = firConv->convertType(memRefTy);
      auto undef = aiir::LLVM::UndefOp::create(rewriter, loc, llvmMemRefTy);
      auto insert =
          aiir::LLVM::InsertValueOp::create(rewriter, loc, undef, basePtr, 1);
      rewriter.replaceOp(convert, insert);
      return aiir::success();
    }

    if (auto memRefTy = aiir::dyn_cast<aiir::MemRefType>(fromFirTy)) {
      // Legalize conversions *from* memref descriptors to pointer-like values
      // by extracting the underlying buffer pointer from the descriptor.
      aiir::Location loc = convert.getLoc();
      aiir::Value base = adaptor.getValue();
      auto alignedPtr =
          aiir::LLVM::ExtractValueOp::create(rewriter, loc, base, 1);
      auto offset = aiir::LLVM::ExtractValueOp::create(rewriter, loc, base, 2);
      aiir::Type elementType =
          this->getTypeConverter()->convertType(memRefTy.getElementType());
      auto gepOp = aiir::LLVM::GEPOp::create(rewriter, loc,
                                             alignedPtr.getType(), elementType,
                                             alignedPtr, offset.getResult());
      rewriter.replaceOp(convert, gepOp);
      return aiir::success();
    }

    auto fromTy = convertType(fromFirTy);
    auto toTy = convertType(toFirTy);
    aiir::Value op0 = adaptor.getOperands()[0];

    if (fromFirTy == toFirTy) {
      rewriter.replaceOp(convert, op0);
      return aiir::success();
    }

    auto loc = convert.getLoc();
    auto i1Type = aiir::IntegerType::get(convert.getContext(), 1);

    if (aiir::isa<fir::RecordType>(toFirTy)) {
      // Convert to compatible BIND(C) record type.
      // Double check that the record types are compatible (it should have
      // already been checked by the verifier).
      assert(aiir::cast<fir::RecordType>(fromFirTy).getTypeList() ==
                 aiir::cast<fir::RecordType>(toFirTy).getTypeList() &&
             "incompatible record types");

      auto toStTy = aiir::cast<aiir::LLVM::LLVMStructType>(toTy);
      aiir::Value val = aiir::LLVM::UndefOp::create(rewriter, loc, toStTy);
      auto indexTypeMap = toStTy.getSubelementIndexMap();
      assert(indexTypeMap.has_value() && "invalid record type");

      for (auto [attr, type] : indexTypeMap.value()) {
        int64_t index = aiir::cast<aiir::IntegerAttr>(attr).getInt();
        auto extVal =
            aiir::LLVM::ExtractValueOp::create(rewriter, loc, op0, index);
        val = aiir::LLVM::InsertValueOp::create(rewriter, loc, val, extVal,
                                                index);
      }

      rewriter.replaceOp(convert, val);
      return aiir::success();
    }

    if (aiir::isa<fir::LogicalType>(fromFirTy) ||
        aiir::isa<fir::LogicalType>(toFirTy)) {
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
      if (!aiir::isa<aiir::IntegerType>(fromTy) ||
          !aiir::isa<aiir::IntegerType>(toTy))
        return aiir::emitError(loc)
               << "unsupported types for logical conversion: " << fromTy
               << " -> " << toTy;

      // Do folding for constant inputs.
      if (auto constVal = fir::getIntIfConstant(op0)) {
        aiir::Value normVal =
            fir::genConstantIndex(loc, toTy, rewriter, *constVal ? 1 : 0);
        rewriter.replaceOp(convert, normVal);
        return aiir::success();
      }

      // If the input is i1, then we can just zero extend it, and
      // the result will be normalized.
      if (fromTy == i1Type) {
        rewriter.replaceOpWithNewOp<aiir::LLVM::ZExtOp>(convert, toTy, op0);
        return aiir::success();
      }

      // Compare the input with zero.
      aiir::Value zero = fir::genConstantIndex(loc, fromTy, rewriter, 0);
      auto isTrue = aiir::LLVM::ICmpOp::create(
          rewriter, loc, aiir::LLVM::ICmpPredicate::ne, op0, zero);

      // Zero extend the i1 isTrue result to the required type (unless it is i1
      // itself).
      if (toTy != i1Type)
        rewriter.replaceOpWithNewOp<aiir::LLVM::ZExtOp>(convert, toTy, isTrue);
      else
        rewriter.replaceOp(convert, isTrue.getResult());

      return aiir::success();
    }

    if (fromTy == toTy) {
      rewriter.replaceOp(convert, op0);
      return aiir::success();
    }
    auto convertFpToFp = [&](aiir::Value val, unsigned fromBits,
                             unsigned toBits, aiir::Type toTy) -> aiir::Value {
      if (fromBits == toBits) {
        // TODO: Converting between two floating-point representations with the
        // same bitwidth is not allowed for now.
        aiir::emitError(loc,
                        "cannot implicitly convert between two floating-point "
                        "representations of the same bitwidth");
        return {};
      }
      if (fromBits > toBits)
        return aiir::LLVM::FPTruncOp::create(rewriter, loc, toTy, val);
      return aiir::LLVM::FPExtOp::create(rewriter, loc, toTy, val);
    };
    // Complex to complex conversion.
    if (fir::isa_complex(fromFirTy) && fir::isa_complex(toFirTy)) {
      // Special case: handle the conversion of a complex such that both the
      // real and imaginary parts are converted together.
      auto ty = convertType(getComplexEleTy(convert.getValue().getType()));
      auto rp = aiir::LLVM::ExtractValueOp::create(rewriter, loc, op0, 0);
      auto ip = aiir::LLVM::ExtractValueOp::create(rewriter, loc, op0, 1);
      auto nt = convertType(getComplexEleTy(convert.getRes().getType()));
      auto fromBits = aiir::LLVM::getPrimitiveTypeSizeInBits(ty);
      auto toBits = aiir::LLVM::getPrimitiveTypeSizeInBits(nt);
      auto rc = convertFpToFp(rp, fromBits, toBits, nt);
      auto ic = convertFpToFp(ip, fromBits, toBits, nt);
      auto un = aiir::LLVM::UndefOp::create(rewriter, loc, toTy);
      llvm::SmallVector<int64_t> pos{0};
      auto i1 = aiir::LLVM::InsertValueOp::create(rewriter, loc, un, rc, pos);
      rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(convert, i1, ic,
                                                             1);
      return aiir::success();
    }

    // Floating point to floating point conversion.
    if (isFloatingPointTy(fromTy)) {
      if (isFloatingPointTy(toTy)) {
        auto fromBits = aiir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = aiir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        auto v = convertFpToFp(op0, fromBits, toBits, toTy);
        rewriter.replaceOp(convert, v);
        return aiir::success();
      }
      if (aiir::isa<aiir::IntegerType>(toTy)) {
        // NOTE: We are checking the fir type here because toTy is an LLVM type
        // which is signless, and we need to use the intrinsic that matches the
        // sign of the output in fir.
        if (toFirTy.isUnsignedInteger()) {
          auto intrinsicName =
              aiir::StringAttr::get(convert.getContext(), "llvm.fptoui.sat");
          rewriter.replaceOpWithNewOp<aiir::LLVM::CallIntrinsicOp>(
              convert, toTy, intrinsicName, op0);
        } else {
          auto intrinsicName =
              aiir::StringAttr::get(convert.getContext(), "llvm.fptosi.sat");
          rewriter.replaceOpWithNewOp<aiir::LLVM::CallIntrinsicOp>(
              convert, toTy, intrinsicName, op0);
        }
        return aiir::success();
      }
    } else if (aiir::isa<aiir::IntegerType>(fromTy)) {
      // Integer to integer conversion.
      if (aiir::isa<aiir::IntegerType>(toTy)) {
        auto fromBits = aiir::LLVM::getPrimitiveTypeSizeInBits(fromTy);
        auto toBits = aiir::LLVM::getPrimitiveTypeSizeInBits(toTy);
        assert(fromBits != toBits);
        if (fromBits > toBits) {
          rewriter.replaceOpWithNewOp<aiir::LLVM::TruncOp>(convert, toTy, op0);
          return aiir::success();
        }
        if (fromFirTy == i1Type || fromFirTy.isUnsignedInteger()) {
          rewriter.replaceOpWithNewOp<aiir::LLVM::ZExtOp>(convert, toTy, op0);
          return aiir::success();
        }
        rewriter.replaceOpWithNewOp<aiir::LLVM::SExtOp>(convert, toTy, op0);
        return aiir::success();
      }
      // Integer to floating point conversion.
      if (isFloatingPointTy(toTy)) {
        if (fromTy.isUnsignedInteger())
          rewriter.replaceOpWithNewOp<aiir::LLVM::UIToFPOp>(convert, toTy, op0);
        else
          rewriter.replaceOpWithNewOp<aiir::LLVM::SIToFPOp>(convert, toTy, op0);
        return aiir::success();
      }
      // Integer to pointer conversion.
      if (aiir::isa<aiir::LLVM::LLVMPointerType>(toTy)) {
        rewriter.replaceOpWithNewOp<aiir::LLVM::IntToPtrOp>(convert, toTy, op0);
        return aiir::success();
      }
    } else if (aiir::isa<aiir::LLVM::LLVMPointerType>(fromTy)) {
      // Pointer to integer conversion.
      if (aiir::isa<aiir::IntegerType>(toTy)) {
        rewriter.replaceOpWithNewOp<aiir::LLVM::PtrToIntOp>(convert, toTy, op0);
        return aiir::success();
      }
      // Pointer to pointer conversion.
      if (aiir::isa<aiir::LLVM::LLVMPointerType>(toTy)) {
        rewriter.replaceOpWithNewOp<aiir::LLVM::BitcastOp>(convert, toTy, op0);
        return aiir::success();
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return aiir::success();
  }
};

/// `fir.dt_entry` operation has no specific CodeGen. The operation is only used
/// to carry information during FIR to FIR passes.
struct DTEntryOpConversion : public fir::FIROpConversion<fir::DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::DTEntryOp op, OpAdaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return aiir::success();
  }
};

/// Lower `fir.global_len` operation.
struct GlobalLenOpConversion : public fir::FIROpConversion<fir::GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::GlobalLenOp globalLen, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    TODO(globalLen.getLoc(), "fir.global_len codegen");
    return aiir::failure();
  }
};

/// Lower fir.len_param_index
struct LenParamIndexOpConversion
    : public fir::FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  llvm::LogicalResult
  matchAndRewrite(fir::LenParamIndexOp lenp, OpAdaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();

    aiir::Value charBuffer = operands[0];
    aiir::Value charBufferLen = operands[1];

    aiir::Location loc = emboxChar.getLoc();
    aiir::Type llvmStructTy = convertType(emboxChar.getType());
    auto llvmStruct = aiir::LLVM::UndefOp::create(rewriter, loc, llvmStructTy);

    aiir::Type lenTy =
        aiir::cast<aiir::LLVM::LLVMStructType>(llvmStructTy).getBody()[1];
    aiir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, charBufferLen);

    aiir::Type addrTy =
        aiir::cast<aiir::LLVM::LLVMStructType>(llvmStructTy).getBody()[0];
    if (addrTy != charBuffer.getType())
      charBuffer =
          aiir::LLVM::BitcastOp::create(rewriter, loc, addrTy, charBuffer);

    llvm::SmallVector<int64_t> pos{0};
    auto insertBufferOp = aiir::LLVM::InsertValueOp::create(
        rewriter, loc, llvmStruct, charBuffer, pos);
    rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(
        emboxChar, insertBufferOp, lenAfterCast, 1);

    return aiir::success();
  }
};
} // namespace

template <typename ModuleOp>
static aiir::SymbolRefAttr
getMallocInModule(ModuleOp mod, fir::AllocMemOp op,
                  aiir::ConversionPatternRewriter &rewriter,
                  aiir::Type indexType) {
  static constexpr char mallocName[] = "malloc";
  if (auto mallocFunc =
          mod.template lookupSymbol<aiir::LLVM::LLVMFuncOp>(mallocName))
    return aiir::SymbolRefAttr::get(mallocFunc);
  if (auto userMalloc =
          mod.template lookupSymbol<aiir::func::FuncOp>(mallocName))
    return aiir::SymbolRefAttr::get(userMalloc);

  aiir::OpBuilder moduleBuilder(mod.getBodyRegion());
  auto mallocDecl = aiir::LLVM::LLVMFuncOp::create(
      moduleBuilder, op.getLoc(), mallocName,
      aiir::LLVM::LLVMFunctionType::get(getLlvmPtrType(op.getContext()),
                                        indexType,
                                        /*isVarArg=*/false));
  return aiir::SymbolRefAttr::get(mallocDecl);
}

/// Return the LLVMFuncOp corresponding to the standard malloc call.
static aiir::SymbolRefAttr getMalloc(fir::AllocMemOp op,
                                     aiir::ConversionPatternRewriter &rewriter,
                                     aiir::Type indexType) {
  if (auto mod = op->getParentOfType<aiir::gpu::GPUModuleOp>())
    return getMallocInModule(mod, op, rewriter, indexType);
  auto mod = op->getParentOfType<aiir::ModuleOp>();
  return getMallocInModule(mod, op, rewriter, indexType);
}

/// Return value of the stride in bytes between adjacent elements
/// of LLVM type \p llTy. The result is returned as a value of
/// \p idxTy integer type.
static aiir::Value
genTypeStrideInBytes(aiir::Location loc, aiir::Type idxTy,
                     aiir::ConversionPatternRewriter &rewriter, aiir::Type llTy,
                     const aiir::DataLayout &dataLayout) {
  // Create a pointer type and use computeElementDistance().
  return fir::computeElementDistance(loc, llTy, idxTy, rewriter, dataLayout);
}

namespace {
/// Lower a `fir.allocmem` instruction into `llvm.call @malloc`
struct AllocMemOpConversion : public fir::FIROpConversion<fir::AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AllocMemOp heap, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type heapTy = heap.getType();
    aiir::Location loc = heap.getLoc();
    auto ity = lowerTy().indexType();
    aiir::Type dataTy = fir::unwrapRefType(heapTy);
    aiir::Type llvmObjectTy = convertObjectType(dataTy);
    if (fir::isRecordWithTypeParameters(fir::unwrapSequenceType(dataTy)))
      TODO(loc, "fir.allocmem codegen of derived type with length parameters");
    aiir::Value size = genTypeSizeInBytes(loc, ity, rewriter, llvmObjectTy);
    if (auto scaleSize =
            fir::genAllocationScaleSize(loc, heap.getInType(), ity, rewriter))
      size = aiir::LLVM::MulOp::create(rewriter, loc, ity, size, scaleSize);
    for (aiir::Value opnd : adaptor.getOperands())
      size = aiir::LLVM::MulOp::create(rewriter, loc, ity, size,
                                       integerCast(loc, rewriter, ity, opnd));

    // As the return value of malloc(0) is implementation defined, allocate one
    // byte to ensure the allocation status being true. This behavior aligns to
    // what the runtime has.
    aiir::Value zero = fir::genConstantIndex(loc, ity, rewriter, 0);
    aiir::Value one = fir::genConstantIndex(loc, ity, rewriter, 1);
    aiir::Value cmp = aiir::LLVM::ICmpOp::create(
        rewriter, loc, aiir::LLVM::ICmpPredicate::sgt, size, zero);
    size = aiir::LLVM::SelectOp::create(rewriter, loc, cmp, size, one);

    auto mallocTyWidth = lowerTy().getIndexTypeBitwidth();
    auto mallocTy =
        aiir::IntegerType::get(rewriter.getContext(), mallocTyWidth);
    if (mallocTyWidth != ity.getIntOrFloatBitWidth())
      size = integerCast(loc, rewriter, mallocTy, size);
    heap->setAttr("callee", getMalloc(heap, rewriter, mallocTy));
    rewriter.replaceOpWithNewOp<aiir::LLVM::CallOp>(
        heap, ::getLlvmPtrType(heap.getContext()), size,
        addLLVMOpBundleAttrs(rewriter, heap->getAttrs(), 1));
    return aiir::success();
  }

  /// Compute the allocation size in bytes of the element type of
  /// \p llTy pointer type. The result is returned as a value of \p idxTy
  /// integer type.
  aiir::Value genTypeSizeInBytes(aiir::Location loc, aiir::Type idxTy,
                                 aiir::ConversionPatternRewriter &rewriter,
                                 aiir::Type llTy) const {
    return fir::computeElementDistance(loc, llTy, idxTy, rewriter,
                                       getDataLayout());
  }
};
} // namespace

/// Return the LLVMFuncOp corresponding to the standard free call.
template <typename ModuleOp>
static aiir::SymbolRefAttr
getFreeInModule(ModuleOp mod, fir::FreeMemOp op,
                aiir::ConversionPatternRewriter &rewriter) {
  static constexpr char freeName[] = "free";
  // Check if free already defined in the module.
  if (auto freeFunc =
          mod.template lookupSymbol<aiir::LLVM::LLVMFuncOp>(freeName))
    return aiir::SymbolRefAttr::get(freeFunc);
  if (auto freeDefinedByUser =
          mod.template lookupSymbol<aiir::func::FuncOp>(freeName))
    return aiir::SymbolRefAttr::get(freeDefinedByUser);
  // Create llvm declaration for free.
  aiir::OpBuilder moduleBuilder(mod.getBodyRegion());
  auto voidType = aiir::LLVM::LLVMVoidType::get(op.getContext());
  auto freeDecl = aiir::LLVM::LLVMFuncOp::create(
      moduleBuilder, rewriter.getUnknownLoc(), freeName,
      aiir::LLVM::LLVMFunctionType::get(voidType,
                                        getLlvmPtrType(op.getContext()),
                                        /*isVarArg=*/false));
  return aiir::SymbolRefAttr::get(freeDecl);
}

static aiir::SymbolRefAttr getFree(fir::FreeMemOp op,
                                   aiir::ConversionPatternRewriter &rewriter) {
  if (auto mod = op->getParentOfType<aiir::gpu::GPUModuleOp>())
    return getFreeInModule(mod, op, rewriter);
  auto mod = op->getParentOfType<aiir::ModuleOp>();
  return getFreeInModule(mod, op, rewriter);
}

static unsigned getDimension(aiir::LLVM::LLVMArrayType ty) {
  unsigned result = 1;
  for (auto eleTy =
           aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(ty.getElementType());
       eleTy; eleTy = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = freemem.getLoc();
    freemem->setAttr("callee", getFree(freemem, rewriter));
    aiir::LLVM::CallOp::create(
        rewriter, loc, aiir::TypeRange{},
        aiir::ValueRange{adaptor.getHeapref()},
        addLLVMOpBundleAttrs(rewriter, freemem->getAttrs(), 1));
    rewriter.eraseOp(freemem);
    return aiir::success();
  }
};
} // namespace

// Convert subcomponent array indices from column-major to row-major ordering.
static llvm::SmallVector<aiir::Value>
convertSubcomponentIndices(aiir::Location loc, aiir::Type eleTy,
                           aiir::ValueRange indices,
                           aiir::Type *retTy = nullptr) {
  llvm::SmallVector<aiir::Value> result;
  llvm::SmallVector<aiir::Value> arrayIndices;

  auto appendArrayIndices = [&] {
    if (arrayIndices.empty())
      return;
    std::reverse(arrayIndices.begin(), arrayIndices.end());
    result.append(arrayIndices.begin(), arrayIndices.end());
    arrayIndices.clear();
  };

  for (aiir::Value index : indices) {
    // Component indices can be field index to select a component, or array
    // index, to select an element in an array component.
    if (auto structTy = aiir::dyn_cast<aiir::LLVM::LLVMStructType>(eleTy)) {
      std::int64_t cstIndex = getConstantIntValue(index);
      assert(cstIndex < (int64_t)structTy.getBody().size() &&
             "out-of-bounds struct field index");
      eleTy = structTy.getBody()[cstIndex];
      appendArrayIndices();
      result.push_back(index);
    } else if (auto arrayTy =
                   aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(eleTy)) {
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

static aiir::Value genSourceFile(aiir::Location loc, aiir::ModuleOp mod,
                                 aiir::ConversionPatternRewriter &rewriter) {
  auto ptrTy = aiir::LLVM::LLVMPointerType::get(rewriter.getContext());
  if (auto flc = aiir::dyn_cast<aiir::FileLineColLoc>(loc)) {
    auto fn = flc.getFilename().str() + '\0';
    std::string globalName = fir::factory::uniqueCGIdent("cl", fn);

    if (auto g = mod.lookupSymbol<fir::GlobalOp>(globalName)) {
      return aiir::LLVM::AddressOfOp::create(rewriter, loc, ptrTy, g.getName());
    } else if (auto g = mod.lookupSymbol<aiir::LLVM::GlobalOp>(globalName)) {
      return aiir::LLVM::AddressOfOp::create(rewriter, loc, ptrTy, g.getName());
    }

    auto crtInsPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(mod.getBody(), mod.getBody()->end());
    auto arrayTy = aiir::LLVM::LLVMArrayType::get(
        aiir::IntegerType::get(rewriter.getContext(), 8), fn.size());
    aiir::LLVM::GlobalOp globalOp = aiir::LLVM::GlobalOp::create(
        rewriter, loc, arrayTy, /*constant=*/true,
        aiir::LLVM::Linkage::Linkonce, globalName, aiir::Attribute());

    aiir::Region &region = globalOp.getInitializerRegion();
    aiir::Block *block = rewriter.createBlock(&region);
    rewriter.setInsertionPoint(block, block->begin());
    aiir::Value constValue = aiir::LLVM::ConstantOp::create(
        rewriter, loc, arrayTy, rewriter.getStringAttr(fn));
    aiir::LLVM::ReturnOp::create(rewriter, loc, constValue);
    rewriter.restoreInsertionPoint(crtInsPt);
    return aiir::LLVM::AddressOfOp::create(rewriter, loc, ptrTy,
                                           globalOp.getName());
  }
  return aiir::LLVM::ZeroOp::create(rewriter, loc, ptrTy);
}

static aiir::Value genSourceLine(aiir::Location loc,
                                 aiir::ConversionPatternRewriter &rewriter) {
  if (auto flc = aiir::dyn_cast<aiir::FileLineColLoc>(loc))
    return aiir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                          flc.getLine());
  return aiir::LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                        0);
}

static aiir::Value
genCUFAllocDescriptor(aiir::Location loc,
                      aiir::ConversionPatternRewriter &rewriter,
                      aiir::ModuleOp mod, fir::BaseBoxType boxTy,
                      const fir::LLVMTypeConverter &typeConverter) {
  std::optional<aiir::DataLayout> dl =
      fir::support::getOrSetAIIRDataLayout(mod, /*allowDefaultLayout=*/true);
  if (!dl)
    aiir::emitError(mod.getLoc(),
                    "module operation must carry a data layout attribute "
                    "to generate llvm IR from FIR");

  aiir::Value sourceFile = genSourceFile(loc, mod, rewriter);
  aiir::Value sourceLine = genSourceLine(loc, rewriter);

  aiir::AIIRContext *ctx = mod.getContext();

  aiir::LLVM::LLVMPointerType llvmPointerType =
      aiir::LLVM::LLVMPointerType::get(ctx);
  aiir::Type llvmInt32Type = aiir::IntegerType::get(ctx, 32);
  aiir::Type llvmIntPtrType =
      aiir::IntegerType::get(ctx, typeConverter.getPointerBitwidth(0));
  auto fctTy = aiir::LLVM::LLVMFunctionType::get(
      llvmPointerType, {llvmIntPtrType, llvmPointerType, llvmInt32Type});

  auto llvmFunc = mod.lookupSymbol<aiir::LLVM::LLVMFuncOp>(
      RTNAME_STRING(CUFAllocDescriptor));
  auto funcFunc =
      mod.lookupSymbol<aiir::func::FuncOp>(RTNAME_STRING(CUFAllocDescriptor));
  if (!llvmFunc && !funcFunc) {
    auto builder = aiir::OpBuilder::atBlockEnd(mod.getBody());
    aiir::LLVM::LLVMFuncOp::create(builder, loc,
                                   RTNAME_STRING(CUFAllocDescriptor), fctTy);
  }

  aiir::Type structTy = typeConverter.convertBoxTypeAsStruct(boxTy);
  std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
  aiir::Value sizeInBytes =
      fir::genConstantIndex(loc, llvmIntPtrType, rewriter, boxSize);
  llvm::SmallVector args = {sizeInBytes, sourceFile, sourceLine};
  return aiir::LLVM::CallOp::create(rewriter, loc, fctTy,
                                    RTNAME_STRING(CUFAllocDescriptor), args)
      .getResult();
}

/// Get the address of the type descriptor global variable that was created by
/// lowering for derived type \p recType.
template <typename ModOpTy>
static aiir::Value
getTypeDescriptor(ModOpTy mod, aiir::ConversionPatternRewriter &rewriter,
                  aiir::Location loc, fir::RecordType recType,
                  const fir::FIRToLLVMPassOptions &options) {
  std::string name =
      options.typeDescriptorsRenamedForAssembly
          ? fir::NameUniquer::getTypeDescriptorAssemblyName(recType.getName())
          : fir::NameUniquer::getTypeDescriptorName(recType.getName());
  aiir::Type llvmPtrTy = ::getLlvmPtrType(mod.getContext());
  aiir::DataLayout dataLayout(mod);
  if (auto global = mod.template lookupSymbol<fir::GlobalOp>(name))
    return replaceWithAddrOfOrASCast(
        rewriter, loc, fir::factory::getGlobalAddressSpace(&dataLayout),
        fir::factory::getProgramAddressSpace(&dataLayout), global.getSymName(),
        llvmPtrTy);
  // The global may have already been translated to LLVM.
  if (auto global = mod.template lookupSymbol<aiir::LLVM::GlobalOp>(name))
    return replaceWithAddrOfOrASCast(
        rewriter, loc, global.getAddrSpace(),
        fir::factory::getProgramAddressSpace(&dataLayout), global.getSymName(),
        llvmPtrTy);
  // Type info derived types do not have type descriptors since they are the
  // types defining type descriptors.
  if (options.ignoreMissingTypeDescriptors ||
      fir::NameUniquer::belongsToModule(
          name, Fortran::semantics::typeInfoBuiltinModule))
    return aiir::LLVM::ZeroOp::create(rewriter, loc, llvmPtrTy);

  if (!options.skipExternalRttiDefinition)
    fir::emitFatalError(loc,
                        "runtime derived type info descriptor was not "
                        "generated and skipExternalRttiDefinition and "
                        "ignoreMissingTypeDescriptors options are not set");

  // Rtti for a derived type defined in another compilation unit and for which
  // rtti was not defined in lowering because of the skipExternalRttiDefinition
  // option. Generate the object declaration now.
  auto insertPt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPoint(mod.getBody(), mod.getBody()->end());
  aiir::LLVM::GlobalOp global = aiir::LLVM::GlobalOp::create(
      rewriter, loc, llvmPtrTy, /*constant=*/true,
      aiir::LLVM::Linkage::External, name, aiir::Attribute());
  rewriter.restoreInsertionPoint(insertPt);
  return aiir::LLVM::AddressOfOp::create(rewriter, loc, llvmPtrTy,
                                         global.getSymName());
}

/// Common base class for embox to descriptor conversion.
template <typename OP>
struct EmboxCommonConversion : public fir::FIROpConversion<OP> {
  using fir::FIROpConversion<OP>::FIROpConversion;
  using TypePair = typename fir::FIROpConversion<OP>::TypePair;

  static int getCFIAttr(fir::BaseBoxType boxTy) {
    auto eleTy = boxTy.getEleTy();
    if (aiir::isa<fir::PointerType>(eleTy))
      return CFI_attribute_pointer;
    if (aiir::isa<fir::HeapType>(eleTy))
      return CFI_attribute_allocatable;
    return CFI_attribute_other;
  }

  aiir::Value getCharacterByteSize(aiir::Location loc,
                                   aiir::ConversionPatternRewriter &rewriter,
                                   fir::CharacterType charTy,
                                   aiir::ValueRange lenParams) const {
    auto i64Ty = aiir::IntegerType::get(rewriter.getContext(), 64);
    aiir::Value size = genTypeStrideInBytes(
        loc, i64Ty, rewriter, this->convertType(charTy), this->getDataLayout());
    if (charTy.hasConstantLen())
      return size; // Length accounted for in the genTypeStrideInBytes GEP.
    // Otherwise,  multiply the single character size by the length.
    assert(!lenParams.empty());
    auto len64 = fir::FIROpConversion<OP>::integerCast(loc, rewriter, i64Ty,
                                                       lenParams.back());
    return aiir::LLVM::MulOp::create(rewriter, loc, i64Ty, size, len64);
  }

  // Get the element size and CFI type code of the boxed value.
  std::tuple<aiir::Value, aiir::Value> getSizeAndTypeCode(
      aiir::Location loc, aiir::ConversionPatternRewriter &rewriter,
      aiir::Type boxEleTy, aiir::ValueRange lenParams = {}) const {
    const aiir::DataLayout &dataLayout = this->getDataLayout();
    auto i64Ty = aiir::IntegerType::get(rewriter.getContext(), 64);
    if (auto eleTy = fir::dyn_cast_ptrEleTy(boxEleTy))
      boxEleTy = eleTy;
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(boxEleTy))
      return getSizeAndTypeCode(loc, rewriter, seqTy.getEleTy(), lenParams);
    if (aiir::isa<aiir::NoneType>(
            boxEleTy)) // unlimited polymorphic or assumed type
      return {aiir::LLVM::ConstantOp::create(rewriter, loc, i64Ty, 0),
              this->genConstantOffset(loc, rewriter, CFI_type_other)};
    aiir::Value typeCodeVal = this->genConstantOffset(
        loc, rewriter,
        fir::getTypeCode(boxEleTy, this->lowerTy().getKindMap()));
    if (fir::isa_integer(boxEleTy) ||
        aiir::dyn_cast<fir::LogicalType>(boxEleTy) || fir::isa_real(boxEleTy) ||
        fir::isa_complex(boxEleTy))
      return {genTypeStrideInBytes(loc, i64Ty, rewriter,
                                   this->convertType(boxEleTy), dataLayout),
              typeCodeVal};
    if (auto charTy = aiir::dyn_cast<fir::CharacterType>(boxEleTy))
      return {getCharacterByteSize(loc, rewriter, charTy, lenParams),
              typeCodeVal};
    if (fir::isa_ref_type(boxEleTy)) {
      auto ptrTy = ::getLlvmPtrType(rewriter.getContext());
      return {genTypeStrideInBytes(loc, i64Ty, rewriter, ptrTy, dataLayout),
              typeCodeVal};
    }
    if (aiir::isa<fir::RecordType>(boxEleTy))
      return {genTypeStrideInBytes(loc, i64Ty, rewriter,
                                   this->convertType(boxEleTy), dataLayout),
              typeCodeVal};
    fir::emitFatalError(loc, "unhandled type in fir.box code generation");
  }

  /// Basic pattern to write a field in the descriptor
  aiir::Value insertField(aiir::ConversionPatternRewriter &rewriter,
                          aiir::Location loc, aiir::Value dest,
                          llvm::ArrayRef<std::int64_t> fldIndexes,
                          aiir::Value value, bool bitcast = false) const {
    auto boxTy = dest.getType();
    auto fldTy = this->getBoxEleTy(boxTy, fldIndexes);
    if (!bitcast)
      value = this->integerCast(loc, rewriter, fldTy, value);
    // bitcast are no-ops with LLVM opaque pointers.
    return aiir::LLVM::InsertValueOp::create(rewriter, loc, dest, value,
                                             fldIndexes);
  }

  inline aiir::Value
  insertBaseAddress(aiir::ConversionPatternRewriter &rewriter,
                    aiir::Location loc, aiir::Value dest,
                    aiir::Value base) const {
    return insertField(rewriter, loc, dest, {kAddrPosInBox}, base,
                       /*bitCast=*/true);
  }

  inline aiir::Value insertLowerBound(aiir::ConversionPatternRewriter &rewriter,
                                      aiir::Location loc, aiir::Value dest,
                                      unsigned dim, aiir::Value lb) const {
    return insertField(rewriter, loc, dest,
                       {kDimsPosInBox, dim, kDimLowerBoundPos}, lb);
  }

  inline aiir::Value insertExtent(aiir::ConversionPatternRewriter &rewriter,
                                  aiir::Location loc, aiir::Value dest,
                                  unsigned dim, aiir::Value extent) const {
    return insertField(rewriter, loc, dest, {kDimsPosInBox, dim, kDimExtentPos},
                       extent);
  }

  inline aiir::Value insertStride(aiir::ConversionPatternRewriter &rewriter,
                                  aiir::Location loc, aiir::Value dest,
                                  unsigned dim, aiir::Value stride) const {
    return insertField(rewriter, loc, dest, {kDimsPosInBox, dim, kDimStridePos},
                       stride);
  }

  template <typename ModOpTy>
  aiir::Value populateDescriptor(aiir::Location loc, ModOpTy mod,
                                 fir::BaseBoxType boxTy, aiir::Type inputType,
                                 aiir::ConversionPatternRewriter &rewriter,
                                 unsigned rank, aiir::Value eleSize,
                                 aiir::Value cfiTy, aiir::Value typeDesc,
                                 int allocatorIdx = kDefaultAllocator,
                                 aiir::Value extraField = {}) const {
    auto llvmBoxTy = this->lowerTy().convertBoxTypeAsStruct(boxTy, rank);
    bool isUnlimitedPolymorphic = fir::isUnlimitedPolymorphicType(boxTy);
    bool useInputType = fir::isPolymorphicType(boxTy) || isUnlimitedPolymorphic;
    aiir::Value descriptor =
        aiir::LLVM::UndefOp::create(rewriter, loc, llvmBoxTy);
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
        auto maskAttr = aiir::IntegerAttr::get(
            rewriter.getIntegerType(8, /*isSigned=*/false),
            llvm::APInt(8, (uint64_t)_CFI_ADDENDUM_FLAG, /*isSigned=*/false));
        aiir::LLVM::ConstantOp mask = aiir::LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI8Type(), maskAttr);
        extraField = aiir::LLVM::OrOp::create(rewriter, loc, extraField, mask);
      } else {
        auto maskAttr = aiir::IntegerAttr::get(
            rewriter.getIntegerType(8, /*isSigned=*/false),
            llvm::APInt(8, (uint64_t)~_CFI_ADDENDUM_FLAG, /*isSigned=*/true));
        aiir::LLVM::ConstantOp mask = aiir::LLVM::ConstantOp::create(
            rewriter, loc, rewriter.getI8Type(), maskAttr);
        extraField = aiir::LLVM::AndOp::create(rewriter, loc, extraField, mask);
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
          aiir::Type innerType = fir::unwrapInnerType(inputType);
          if (innerType && aiir::isa<fir::RecordType>(innerType)) {
            auto recTy = aiir::dyn_cast<fir::RecordType>(innerType);
            typeDesc =
                getTypeDescriptor(mod, rewriter, loc, recTy, this->options);
          } else {
            // Unlimited polymorphic type descriptor with no record type. Set
            // type descriptor address to a clean state.
            typeDesc = aiir::LLVM::ZeroOp::create(
                rewriter, loc, ::getLlvmPtrType(mod.getContext()));
          }
        } else {
          typeDesc = getTypeDescriptor(
              mod, rewriter, loc, fir::unwrapIfDerived(boxTy), this->options);
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
      aiir::Value zero =
          fir::genConstantIndex(loc, rewriter.getI64Type(), rewriter, 0);
      descriptor = insertField(rewriter, loc, descriptor,
                               {getLenParamFieldId(boxTy), 0}, zero);
    }
    return descriptor;
  }

  // Template used for fir::EmboxOp and fir::cg::XEmboxOp
  template <typename BOX>
  std::tuple<fir::BaseBoxType, aiir::Value, aiir::Value>
  consDescriptorPrefix(BOX box, aiir::Type inputType,
                       aiir::ConversionPatternRewriter &rewriter, unsigned rank,
                       [[maybe_unused]] aiir::ValueRange substrParams,
                       aiir::ValueRange lenParams, aiir::Value sourceBox = {},
                       aiir::Type sourceBoxType = {}) const {
    auto loc = box.getLoc();
    auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(box.getType());
    bool useInputType = fir::isPolymorphicType(boxTy) &&
                        !fir::isUnlimitedPolymorphicType(inputType);
    llvm::SmallVector<aiir::Value> typeparams = lenParams;
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

    aiir::Value typeDesc;
    aiir::Value extraField;
    // When emboxing to a polymorphic box, get the type descriptor, type code
    // and element size from the source box if any.
    if (fir::isPolymorphicType(boxTy) && sourceBox) {
      TypePair sourceBoxTyPair = this->getBoxTypePair(sourceBoxType);
      typeDesc =
          this->loadTypeDescAddress(loc, sourceBoxTyPair, sourceBox, rewriter);
      aiir::Type idxTy = this->lowerTy().indexType();
      eleSize = this->getElementSizeFromBox(loc, idxTy, sourceBoxTyPair,
                                            sourceBox, rewriter);
      cfiTy = this->getValueFromBox(loc, sourceBoxTyPair, sourceBox,
                                    cfiTy.getType(), rewriter, kTypePosInBox);
      extraField =
          this->getExtraFromBox(loc, sourceBoxTyPair, sourceBox, rewriter);
    }

    aiir::Value descriptor;
    if (auto gpuMod = box->template getParentOfType<aiir::gpu::GPUModuleOp>())
      descriptor = populateDescriptor(loc, gpuMod, boxTy, inputType, rewriter,
                                      rank, eleSize, cfiTy, typeDesc,
                                      allocatorIdx, extraField);
    else if (auto mod = box->template getParentOfType<aiir::ModuleOp>())
      descriptor = populateDescriptor(loc, mod, boxTy, inputType, rewriter,
                                      rank, eleSize, cfiTy, typeDesc,
                                      allocatorIdx, extraField);

    return {boxTy, descriptor, eleSize};
  }

  std::tuple<fir::BaseBoxType, aiir::Value, aiir::Value>
  consDescriptorPrefix(fir::cg::XReboxOp box, aiir::Value loweredBox,
                       aiir::ConversionPatternRewriter &rewriter, unsigned rank,
                       aiir::ValueRange substrParams,
                       aiir::ValueRange lenParams,
                       aiir::Value typeDesc = {}) const {
    auto loc = box.getLoc();
    auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(box.getType());
    auto inputBoxTy = aiir::dyn_cast<fir::BaseBoxType>(box.getBox().getType());
    auto inputBoxTyPair = this->getBoxTypePair(inputBoxTy);
    llvm::SmallVector<aiir::Value> typeparams = lenParams;
    if (!box.getSubstr().empty() && fir::hasDynamicSize(boxTy.getEleTy()))
      typeparams.push_back(substrParams[1]);

    auto [eleSize, cfiTy] =
        getSizeAndTypeCode(loc, rewriter, boxTy.getEleTy(), typeparams);

    // Reboxing to a polymorphic entity. eleSize and type code need to
    // be retrieved from the initial box and propagated to the new box.
    // If the initial box has an addendum, the type desc must be propagated as
    // well.
    if (fir::isPolymorphicType(boxTy)) {
      aiir::Type idxTy = this->lowerTy().indexType();
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

    aiir::Value extraField =
        this->getExtraFromBox(loc, inputBoxTyPair, loweredBox, rewriter);

    aiir::Value descriptor;
    if (auto gpuMod = box->template getParentOfType<aiir::gpu::GPUModuleOp>())
      descriptor =
          populateDescriptor(loc, gpuMod, boxTy, box.getBox().getType(),
                             rewriter, rank, eleSize, cfiTy, typeDesc,
                             /*allocatorIdx=*/kDefaultAllocator, extraField);
    else if (auto mod = box->template getParentOfType<aiir::ModuleOp>())
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
  aiir::Value
  genBoxOffsetGep(aiir::ConversionPatternRewriter &rewriter, aiir::Location loc,
                  aiir::Value base, aiir::Type llvmBaseObjectType,
                  aiir::Value outerOffset, aiir::ValueRange cstInteriorIndices,
                  aiir::ValueRange componentIndices,
                  std::optional<aiir::Value> substringOffset) const {
    llvm::SmallVector<aiir::LLVM::GEPArg> gepArgs{outerOffset};
    aiir::Type resultTy = llvmBaseObjectType;
    // Fortran is column major, llvm GEP is row major: reverse the indices here.
    for (aiir::Value interiorIndex : llvm::reverse(cstInteriorIndices)) {
      auto arrayTy = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(resultTy);
      if (!arrayTy)
        fir::emitFatalError(
            loc,
            "corrupted GEP generated being generated in fir.embox/fir.rebox");
      resultTy = arrayTy.getElementType();
      gepArgs.push_back(interiorIndex);
    }
    llvm::SmallVector<aiir::Value> gepIndices =
        convertSubcomponentIndices(loc, resultTy, componentIndices, &resultTy);
    gepArgs.append(gepIndices.begin(), gepIndices.end());
    if (substringOffset) {
      if (auto arrayTy = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(resultTy)) {
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
        aiir::Type outterOffsetTy =
            llvm::cast<aiir::Value>(gepArgs[0]).getType();
        aiir::Value cast =
            this->integerCast(loc, rewriter, outterOffsetTy, *substringOffset);

        gepArgs[0] = aiir::LLVM::AddOp::create(
            rewriter, loc, outterOffsetTy, llvm::cast<aiir::Value>(gepArgs[0]),
            cast);
      }
    }
    aiir::Type llvmPtrTy = ::getLlvmPtrType(resultTy.getContext());
    return aiir::LLVM::GEPOp::create(rewriter, loc, llvmPtrTy,
                                     llvmBaseObjectType, base, gepArgs);
  }

  template <typename BOX>
  void
  getSubcomponentIndices(BOX xbox, aiir::Value memref,
                         aiir::ValueRange operands,
                         aiir::SmallVectorImpl<aiir::Value> &indices) const {
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

  /// If the embox is not in a globalOp body, allocate storage for the box;
  /// store the value inside and return the generated alloca. Return the input
  /// value otherwise.
  aiir::Value
  placeInMemoryIfNotGlobalInit(aiir::ConversionPatternRewriter &rewriter,
                               aiir::Location loc, aiir::Type boxTy,
                               aiir::Value boxValue,
                               bool needDeviceAllocation = false) const {
    if (isInGlobalOp(rewriter))
      return boxValue;
    aiir::Type llvmBoxTy = boxValue.getType();
    aiir::Value storage;
    if (needDeviceAllocation) {
      auto mod = boxValue.getDefiningOp()->getParentOfType<aiir::ModuleOp>();
      auto baseBoxTy = aiir::dyn_cast<fir::BaseBoxType>(boxTy);
      storage =
          genCUFAllocDescriptor(loc, rewriter, mod, baseBoxTy, this->lowerTy());
    } else {
      storage = this->genAllocaAndAddrCastWithType(loc, llvmBoxTy, defaultAlign,
                                                   rewriter);
    }
    auto storeOp =
        aiir::LLVM::StoreOp::create(rewriter, loc, boxValue, storage);
    this->attachTBAATag(storeOp, boxTy, boxTy, nullptr);
    return storage;
  }

  /// Compute the extent of a triplet slice (lb:ub:step).
  aiir::Value computeTripletExtent(aiir::ConversionPatternRewriter &rewriter,
                                   aiir::Location loc, aiir::Value lb,
                                   aiir::Value ub, aiir::Value step,
                                   aiir::Value zero, aiir::Type type) const {
    lb = this->integerCast(loc, rewriter, type, lb);
    ub = this->integerCast(loc, rewriter, type, ub);
    step = this->integerCast(loc, rewriter, type, step);
    zero = this->integerCast(loc, rewriter, type, zero);
    aiir::Value extent = aiir::LLVM::SubOp::create(rewriter, loc, type, ub, lb);
    extent = aiir::LLVM::AddOp::create(rewriter, loc, type, extent, step);
    extent = aiir::LLVM::SDivOp::create(rewriter, loc, type, extent, step);
    // If the resulting extent is negative (`ub-lb` and `step` have different
    // signs), zero must be returned instead.
    auto cmp = aiir::LLVM::ICmpOp::create(
        rewriter, loc, aiir::LLVM::ICmpPredicate::sgt, extent, zero);
    return aiir::LLVM::SelectOp::create(rewriter, loc, cmp, extent, zero);
  }
};

/// Create a generic box on a memory reference. This conversions lowers the
/// abstract box to the appropriate, initialized descriptor.
struct EmboxOpConversion : public EmboxCommonConversion<fir::EmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::EmboxOp embox, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();
    aiir::Value sourceBox;
    aiir::Type sourceBoxType;
    if (embox.getSourceBox()) {
      sourceBox = operands[embox.getSourceBoxOperandIndex()];
      sourceBoxType = embox.getSourceBox().getType();
    }
    assert(!embox.getShape() && "There should be no dims on this embox op");
    auto [boxTy, dest, eleSize] = consDescriptorPrefix(
        embox, fir::unwrapRefType(embox.getMemref().getType()), rewriter,
        /*rank=*/0, /*substrParams=*/aiir::ValueRange{},
        adaptor.getTypeparams(), sourceBox, sourceBoxType);
    dest = insertBaseAddress(rewriter, embox.getLoc(), dest, operands[0]);
    if (fir::isDerivedTypeWithLenParams(boxTy)) {
      TODO(embox.getLoc(),
           "fir.embox codegen of derived with length parameters");
      return aiir::failure();
    }
    auto result =
        placeInMemoryIfNotGlobalInit(rewriter, embox.getLoc(), boxTy, dest);
    rewriter.replaceOp(embox, result);
    return aiir::success();
  }
};

static bool isDeviceAllocation(aiir::Value val, aiir::Value adaptorVal) {
  if (val.getDefiningOp() &&
      val.getDefiningOp()->getParentOfType<aiir::gpu::GPUModuleOp>())
    return false;
  // Check if the global symbol is in the device module.
  if (auto addr = aiir::dyn_cast_or_null<fir::AddrOfOp>(val.getDefiningOp()))
    if (auto gpuMod =
            addr->getParentOfType<aiir::ModuleOp>()
                .lookupSymbol<aiir::gpu::GPUModuleOp>(cudaDeviceModuleName))
      if (gpuMod.lookupSymbol<aiir::LLVM::GlobalOp>(addr.getSymbol()) ||
          gpuMod.lookupSymbol<fir::GlobalOp>(addr.getSymbol()))
        return true;

  if (auto loadOp = aiir::dyn_cast_or_null<fir::LoadOp>(val.getDefiningOp()))
    return isDeviceAllocation(loadOp.getMemref(), {});
  if (auto boxAddrOp =
          aiir::dyn_cast_or_null<fir::BoxAddrOp>(val.getDefiningOp()))
    return isDeviceAllocation(boxAddrOp.getVal(), {});
  if (auto convertOp =
          aiir::dyn_cast_or_null<fir::ConvertOp>(val.getDefiningOp()))
    return isDeviceAllocation(convertOp.getValue(), {});
  if (!val.getDefiningOp() && adaptorVal) {
    if (auto blockArg = llvm::cast<aiir::BlockArgument>(adaptorVal)) {
      if (blockArg.getOwner() && blockArg.getOwner()->getParentOp() &&
          blockArg.getOwner()->isEntryBlock()) {
        if (auto func = aiir::dyn_cast_or_null<aiir::FunctionOpInterface>(
                *blockArg.getOwner()->getParentOp())) {
          auto argAttrs = func.getArgAttrs(blockArg.getArgNumber());
          for (auto attr : argAttrs) {
            if (attr.getName().getValue().ends_with(cuf::getDataAttrName())) {
              auto dataAttr =
                  aiir::dyn_cast<cuf::DataAttributeAttr>(attr.getValue());
              if (dataAttr.getValue() != cuf::DataAttribute::Pinned &&
                  dataAttr.getValue() != cuf::DataAttribute::Unified)
                return true;
            }
          }
        }
      }
    }
  }
  if (auto callOp = aiir::dyn_cast_or_null<fir::CallOp>(val.getDefiningOp()))
    if (callOp.getCallee() &&
        (callOp.getCallee().value().getRootReference().getValue().starts_with(
             RTNAME_STRING(CUFMemAlloc)) ||
         callOp.getCallee().value().getRootReference().getValue().starts_with(
             RTNAME_STRING(CUFAllocDescriptor)) ||
         callOp.getCallee().value().getRootReference().getValue() ==
             "__tgt_acc_get_deviceptr"))
      return true;
  return false;
}

/// Create a generic box on a memory reference.
struct XEmboxOpConversion : public EmboxCommonConversion<fir::cg::XEmboxOp> {
  using EmboxCommonConversion::EmboxCommonConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::cg::XEmboxOp xbox, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();
    aiir::Value sourceBox;
    aiir::Type sourceBoxType;
    if (xbox.getSourceBox()) {
      sourceBox = operands[xbox.getSourceBoxOperandIndex()];
      sourceBoxType = xbox.getSourceBox().getType();
    }
    auto [boxTy, dest, resultEleSize] = consDescriptorPrefix(
        xbox, fir::unwrapRefType(xbox.getMemref().getType()), rewriter,
        xbox.getOutRank(), adaptor.getSubstr(), adaptor.getLenParams(),
        sourceBox, sourceBoxType);
    // Generate the triples in the dims field of the descriptor
    auto i64Ty = aiir::IntegerType::get(xbox.getContext(), 64);
    assert(!xbox.getShape().empty() && "must have a shape");
    unsigned shapeOffset = xbox.getShapeOperandIndex();
    bool hasShift = !xbox.getShift().empty();
    unsigned shiftOffset = xbox.getShiftOperandIndex();
    bool hasSlice = !xbox.getSlice().empty();
    unsigned sliceOffset = xbox.getSliceOperandIndex();
    aiir::Location loc = xbox.getLoc();
    aiir::Value zero = fir::genConstantIndex(loc, i64Ty, rewriter, 0);
    aiir::Value one = fir::genConstantIndex(loc, i64Ty, rewriter, 1);
    aiir::Value prevPtrOff = one;
    aiir::Type eleTy = boxTy.getEleTy();
    const unsigned rank = xbox.getRank();
    llvm::SmallVector<aiir::Value> cstInteriorIndices;
    unsigned constRows = 0;
    aiir::Value ptrOffset = zero;
    aiir::Type memEleTy = fir::dyn_cast_ptrEleTy(xbox.getMemref().getType());
    assert(aiir::isa<fir::SequenceType>(memEleTy));
    auto seqTy = aiir::cast<fir::SequenceType>(memEleTy);
    aiir::Type seqEleTy = seqTy.getEleTy();
    // Adjust the element scaling factor if the element is a dependent type.
    if (fir::hasDynamicSize(seqEleTy)) {
      if (auto charTy = aiir::dyn_cast<fir::CharacterType>(seqEleTy)) {
        // The GEP pointer type decays to llvm.ptr<i[width]>.
        // The scaling factor is the runtime value of the length.
        assert(!adaptor.getLenParams().empty());
        prevPtrOff = FIROpConversion::integerCast(
            loc, rewriter, i64Ty, adaptor.getLenParams().back());
      } else if (aiir::isa<fir::RecordType>(seqEleTy)) {
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
    aiir::Value prevDimByteStride = resultEleSize;
    if (hasSubcomp) {
      // We have a subcomponent. The step value needs to be the number of
      // bytes per element (which is a derived type).
      prevDimByteStride = genTypeStrideInBytes(
          loc, i64Ty, rewriter, convertType(seqEleTy), getDataLayout());
    } else if (hasSubstr) {
      // We have a substring. The step value needs to be the number of bytes
      // per CHARACTER element.
      auto charTy = aiir::cast<fir::CharacterType>(seqEleTy);
      if (fir::hasDynamicSize(charTy)) {
        prevDimByteStride =
            getCharacterByteSize(loc, rewriter, charTy, adaptor.getLenParams());
      } else {
        prevDimByteStride = fir::genConstantIndex(
            loc, i64Ty, rewriter,
            charTy.getLen() * lowerTy().characterBitsize(charTy) / 8);
      }
    }

    // Process the array subspace arguments (shape, shift, etc.), if any,
    // translating everything to values in the descriptor wherever the entity
    // has a dynamic array dimension.
    for (unsigned di = 0, descIdx = 0; di < rank; ++di) {
      aiir::Value extent =
          integerCast(loc, rewriter, i64Ty, operands[shapeOffset]);
      aiir::Value outerExtent = extent;
      bool skipNext = false;
      if (hasSlice) {
        aiir::Value off =
            integerCast(loc, rewriter, i64Ty, operands[sliceOffset]);
        aiir::Value adj = one;
        if (hasShift)
          adj = integerCast(loc, rewriter, i64Ty, operands[shiftOffset]);
        auto ao = aiir::LLVM::SubOp::create(rewriter, loc, i64Ty, off, adj);
        if (constRows > 0) {
          cstInteriorIndices.push_back(ao);
        } else {
          auto dimOff =
              aiir::LLVM::MulOp::create(rewriter, loc, i64Ty, ao, prevPtrOff);
          ptrOffset = aiir::LLVM::AddOp::create(rewriter, loc, i64Ty, dimOff,
                                                ptrOffset);
        }
        if (aiir::isa_and_nonnull<fir::UndefOp>(
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
        aiir::Value lb = zero;
        const bool isaPointerOrAllocatable =
            aiir::isa<fir::PointerType, fir::HeapType>(eleTy);
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
          auto extentIsEmpty = aiir::LLVM::ICmpOp::create(
              rewriter, loc, aiir::LLVM::ICmpPredicate::eq, extent, zero);
          lb = aiir::LLVM::SelectOp::create(rewriter, loc, extentIsEmpty, one,
                                            lb);
        }
        dest = insertLowerBound(rewriter, loc, dest, descIdx, lb);

        dest = insertExtent(rewriter, loc, dest, descIdx, extent);

        // store step (scaled by shaped extent)
        aiir::Value step = prevDimByteStride;
        if (hasSlice) {
          aiir::Value sliceStep =
              integerCast(loc, rewriter, i64Ty, operands[sliceOffset + 2]);
          step =
              aiir::LLVM::MulOp::create(rewriter, loc, i64Ty, step, sliceStep);
        }
        dest = insertStride(rewriter, loc, dest, descIdx, step);
        ++descIdx;
      }

      // compute the stride and offset for the next natural dimension
      prevDimByteStride = aiir::LLVM::MulOp::create(
          rewriter, loc, i64Ty, prevDimByteStride, outerExtent);
      if (constRows == 0)
        prevPtrOff = aiir::LLVM::MulOp::create(rewriter, loc, i64Ty, prevPtrOff,
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
    aiir::Value base = adaptor.getMemref();
    if (hasSlice || hasSubcomp || hasSubstr) {
      // Shift the base address.
      llvm::SmallVector<aiir::Value> fieldIndices;
      std::optional<aiir::Value> substringOffset;
      if (hasSubcomp)
        getSubcomponentIndices(xbox, xbox.getMemref(), operands, fieldIndices);
      if (hasSubstr)
        substringOffset = operands[xbox.getSubstrOperandIndex()];
      aiir::Type llvmBaseType =
          convertType(fir::unwrapRefType(xbox.getMemref().getType()));
      base = genBoxOffsetGep(rewriter, loc, base, llvmBaseType, ptrOffset,
                             cstInteriorIndices, fieldIndices, substringOffset);
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    if (fir::isDerivedTypeWithLenParams(boxTy))
      TODO(loc, "fir.embox codegen of derived with length parameters");
    aiir::Value result = placeInMemoryIfNotGlobalInit(
        rewriter, loc, boxTy, dest,
        isDeviceAllocation(xbox.getMemref(), adaptor.getMemref()));
    rewriter.replaceOp(xbox, result);
    return aiir::success();
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = rebox.getLoc();
    aiir::Type idxTy = lowerTy().indexType();
    aiir::Value loweredBox =
        fixBoxInputInsideGlobalOp(rewriter, adaptor.getBox());
    aiir::ValueRange operands = adaptor.getOperands();

    TypePair inputBoxTyPair = getBoxTypePair(rebox.getBox().getType());

    // Create new descriptor and fill its non-shape related data.
    llvm::SmallVector<aiir::Value, 2> lenParams;
    aiir::Type inputEleTy = getInputEleTy(rebox);
    if (auto charTy = aiir::dyn_cast<fir::CharacterType>(inputEleTy)) {
      if (charTy.hasConstantLen()) {
        aiir::Value len =
            fir::genConstantIndex(loc, idxTy, rewriter, charTy.getLen());
        lenParams.emplace_back(len);
      } else {
        aiir::Value len = getElementSizeFromBox(loc, idxTy, inputBoxTyPair,
                                                loweredBox, rewriter);
        if (charTy.getFKind() != 1) {
          assert(!isInGlobalOp(rewriter) &&
                 "character target in global op must have constant length");
          aiir::Value width =
              fir::genConstantIndex(loc, idxTy, rewriter, charTy.getFKind());
          len = aiir::LLVM::SDivOp::create(rewriter, loc, idxTy, len, width);
        }
        lenParams.emplace_back(len);
      }
    } else if (auto recTy = aiir::dyn_cast<fir::RecordType>(inputEleTy)) {
      if (recTy.getNumLenParams() != 0)
        TODO(loc, "reboxing descriptor of derived type with length parameters");
    }

    // Rebox on polymorphic entities needs to carry over the dynamic type.
    aiir::Value typeDescAddr;
    if (aiir::isa<fir::ClassType>(inputBoxTyPair.fir) &&
        aiir::isa<fir::ClassType>(rebox.getType()))
      typeDescAddr =
          loadTypeDescAddress(loc, inputBoxTyPair, loweredBox, rewriter);

    auto [boxTy, dest, eleSize] =
        consDescriptorPrefix(rebox, loweredBox, rewriter, rebox.getOutRank(),
                             adaptor.getSubstr(), lenParams, typeDescAddr);

    // Read input extents, strides, and base address
    llvm::SmallVector<aiir::Value> inputExtents;
    llvm::SmallVector<aiir::Value> inputStrides;
    const unsigned inputRank = rebox.getRank();
    for (unsigned dim = 0; dim < inputRank; ++dim) {
      llvm::SmallVector<aiir::Value, 3> dimInfo =
          getDimsFromBox(loc, {idxTy, idxTy, idxTy}, inputBoxTyPair, loweredBox,
                         dim, rewriter);
      inputExtents.emplace_back(dimInfo[1]);
      inputStrides.emplace_back(dimInfo[2]);
    }

    aiir::Value baseAddr =
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
                aiir::Type destBoxTy, aiir::Value dest, aiir::Value base,
                aiir::ValueRange lbounds, aiir::ValueRange extents,
                aiir::ValueRange strides,
                aiir::ConversionPatternRewriter &rewriter) const {
    aiir::Location loc = rebox.getLoc();
    aiir::Value zero =
        fir::genConstantIndex(loc, lowerTy().indexType(), rewriter, 0);
    aiir::Value one =
        fir::genConstantIndex(loc, lowerTy().indexType(), rewriter, 1);
    for (auto iter : llvm::enumerate(llvm::zip(extents, strides))) {
      aiir::Value extent = std::get<0>(iter.value());
      unsigned dim = iter.index();
      aiir::Value lb = one;
      if (!lbounds.empty()) {
        lb = integerCast(loc, rewriter, lowerTy().indexType(), lbounds[dim]);
        auto extentIsEmpty = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::eq, extent, zero);
        lb =
            aiir::LLVM::SelectOp::create(rewriter, loc, extentIsEmpty, one, lb);
      };
      dest = insertLowerBound(rewriter, loc, dest, dim, lb);
      dest = insertExtent(rewriter, loc, dest, dim, extent);
      dest = insertStride(rewriter, loc, dest, dim, std::get<1>(iter.value()));
    }
    dest = insertBaseAddress(rewriter, loc, dest, base);
    aiir::Value result = placeInMemoryIfNotGlobalInit(
        rewriter, rebox.getLoc(), destBoxTy, dest,
        isDeviceAllocation(rebox.getBox(), adaptor.getBox()));
    rewriter.replaceOp(rebox, result);
    return aiir::success();
  }

  // Apply slice given the base address, extents and strides of the input box.
  llvm::LogicalResult
  sliceBox(fir::cg::XReboxOp rebox, OpAdaptor adaptor, aiir::Type destBoxTy,
           aiir::Value dest, aiir::Value base, aiir::ValueRange inputExtents,
           aiir::ValueRange inputStrides, aiir::ValueRange operands,
           aiir::ConversionPatternRewriter &rewriter) const {
    aiir::Location loc = rebox.getLoc();
    aiir::Type byteTy = ::getI8Type(rebox.getContext());
    aiir::Type idxTy = lowerTy().indexType();
    aiir::Value zero = fir::genConstantIndex(loc, idxTy, rewriter, 0);
    // Apply subcomponent and substring shift on base address.
    if (!rebox.getSubcomponent().empty() || !rebox.getSubstr().empty()) {
      // Cast to inputEleTy* so that a GEP can be used.
      aiir::Type inputEleTy = getInputEleTy(rebox);
      aiir::Type llvmBaseObjectType = convertType(inputEleTy);
      llvm::SmallVector<aiir::Value> fieldIndices;
      std::optional<aiir::Value> substringOffset;
      if (!rebox.getSubcomponent().empty())
        getSubcomponentIndices(rebox, rebox.getBox(), operands, fieldIndices);
      if (!rebox.getSubstr().empty())
        substringOffset = operands[rebox.getSubstrOperandIndex()];
      base = genBoxOffsetGep(rewriter, loc, base, llvmBaseObjectType, zero,
                             /*cstInteriorIndices=*/{}, fieldIndices,
                             substringOffset);
    }

    if (rebox.getSlice().empty())
      // The array section is of the form array[%component][substring], keep
      // the input array extents and strides.
      return finalizeRebox(rebox, adaptor, destBoxTy, dest, base,
                           /*lbounds*/ {}, inputExtents, inputStrides,
                           rewriter);

    // The slice is of the form array(i:j:k)[%component]. Compute new extents
    // and strides.
    llvm::SmallVector<aiir::Value> slicedExtents;
    llvm::SmallVector<aiir::Value> slicedStrides;
    aiir::Value one = fir::genConstantIndex(loc, idxTy, rewriter, 1);
    const bool sliceHasOrigins = !rebox.getShift().empty();
    unsigned sliceOps = rebox.getSliceOperandIndex();
    unsigned shiftOps = rebox.getShiftOperandIndex();
    auto strideOps = inputStrides.begin();
    const unsigned inputRank = inputStrides.size();
    for (unsigned i = 0; i < inputRank;
         ++i, ++strideOps, ++shiftOps, sliceOps += 3) {
      aiir::Value sliceLb =
          integerCast(loc, rewriter, idxTy, operands[sliceOps]);
      aiir::Value inputStride = *strideOps; // already idxTy
      // Apply origin shift: base += (lb-shift)*input_stride
      aiir::Value sliceOrigin =
          sliceHasOrigins
              ? integerCast(loc, rewriter, idxTy, operands[shiftOps])
              : one;
      aiir::Value diff =
          aiir::LLVM::SubOp::create(rewriter, loc, idxTy, sliceLb, sliceOrigin);
      aiir::Value offset =
          aiir::LLVM::MulOp::create(rewriter, loc, idxTy, diff, inputStride);
      // Strides from the fir.box are in bytes.
      base = genGEP(loc, byteTy, rewriter, base, offset);
      // Apply upper bound and step if this is a triplet. Otherwise, the
      // dimension is dropped and no extents/strides are computed.
      aiir::Value upper = operands[sliceOps + 1];
      const bool isTripletSlice =
          !aiir::isa_and_nonnull<aiir::LLVM::UndefOp>(upper.getDefiningOp());
      if (isTripletSlice) {
        aiir::Value step =
            integerCast(loc, rewriter, idxTy, operands[sliceOps + 2]);
        // extent = ub-lb+step/step
        aiir::Value sliceUb = integerCast(loc, rewriter, idxTy, upper);
        aiir::Value extent = computeTripletExtent(rewriter, loc, sliceLb,
                                                  sliceUb, step, zero, idxTy);
        slicedExtents.emplace_back(extent);
        // stride = step*input_stride
        aiir::Value stride =
            aiir::LLVM::MulOp::create(rewriter, loc, idxTy, step, inputStride);
        slicedStrides.emplace_back(stride);
      }
    }
    return finalizeRebox(rebox, adaptor, destBoxTy, dest, base,
                         /*lbounds*/ {}, slicedExtents, slicedStrides,
                         rewriter);
  }

  /// Apply a new shape to the data described by a box given the base address,
  /// extents and strides of the box.
  llvm::LogicalResult
  reshapeBox(fir::cg::XReboxOp rebox, OpAdaptor adaptor, aiir::Type destBoxTy,
             aiir::Value dest, aiir::Value base, aiir::ValueRange inputExtents,
             aiir::ValueRange inputStrides, aiir::ValueRange operands,
             aiir::ConversionPatternRewriter &rewriter) const {
    aiir::ValueRange reboxShifts{
        operands.begin() + rebox.getShiftOperandIndex(),
        operands.begin() + rebox.getShiftOperandIndex() +
            rebox.getShift().size()};
    if (rebox.getShape().empty()) {
      // Only setting new lower bounds.
      return finalizeRebox(rebox, adaptor, destBoxTy, dest, base, reboxShifts,
                           inputExtents, inputStrides, rewriter);
    }

    aiir::Location loc = rebox.getLoc();

    llvm::SmallVector<aiir::Value> newStrides;
    llvm::SmallVector<aiir::Value> newExtents;
    aiir::Type idxTy = lowerTy().indexType();
    // First stride from input box is kept. The rest is assumed contiguous
    // (it is not possible to reshape otherwise). If the input is scalar,
    // which may be OK if all new extents are ones, the stride does not
    // matter, use one.
    aiir::Value stride = inputStrides.empty()
                             ? fir::genConstantIndex(loc, idxTy, rewriter, 1)
                             : inputStrides[0];
    for (unsigned i = 0; i < rebox.getShape().size(); ++i) {
      aiir::Value rawExtent = operands[rebox.getShapeOperandIndex() + i];
      aiir::Value extent = integerCast(loc, rewriter, idxTy, rawExtent);
      newExtents.emplace_back(extent);
      newStrides.emplace_back(stride);
      // nextStride = extent * stride;
      stride = aiir::LLVM::MulOp::create(rewriter, loc, idxTy, extent, stride);
    }
    return finalizeRebox(rebox, adaptor, destBoxTy, dest, base, reboxShifts,
                         newExtents, newStrides, rewriter);
  }

  /// Return scalar element type of the input box.
  static aiir::Type getInputEleTy(fir::cg::XReboxOp rebox) {
    auto ty = fir::dyn_cast_ptrOrBoxEleTy(rebox.getBox().getType());
    if (auto seqTy = aiir::dyn_cast<fir::SequenceType>(ty))
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    TODO(emboxproc.getLoc(), "fir.emboxproc codegen");
    return aiir::failure();
  }
};

// Code shared between insert_value and extract_value Ops.
struct ValueOpCommon {
  // Translate the arguments pertaining to any multidimensional array to
  // row-major order for LLVM-IR.
  static void toRowMajor(llvm::SmallVectorImpl<int64_t> &indices,
                         aiir::Type ty) {
    assert(ty && "type is null");
    const auto end = indices.size();
    for (std::remove_const_t<decltype(end)> i = 0; i < end; ++i) {
      if (auto seq = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(ty)) {
        const auto dim = getDimension(seq);
        if (dim > 1) {
          auto ub = std::min(i + dim, end);
          std::reverse(indices.begin() + i, indices.begin() + ub);
          i += dim - 1;
        }
        ty = getArrayElementType(seq);
      } else if (auto st = aiir::dyn_cast<aiir::LLVM::LLVMStructType>(ty)) {
        ty = st.getBody()[indices[i]];
      } else {
        llvm_unreachable("index into invalid type");
      }
    }
  }

  static llvm::SmallVector<int64_t>
  collectIndices(aiir::ConversionPatternRewriter &rewriter,
                 aiir::ArrayAttr arrAttr) {
    llvm::SmallVector<int64_t> indices;
    for (auto i = arrAttr.begin(), e = arrAttr.end(); i != e; ++i) {
      if (auto intAttr = aiir::dyn_cast<aiir::IntegerAttr>(*i)) {
        indices.push_back(intAttr.getInt());
      } else {
        auto fieldName = aiir::cast<aiir::StringAttr>(*i).getValue();
        ++i;
        auto ty = aiir::cast<aiir::TypeAttr>(*i).getValue();
        auto index = aiir::cast<fir::RecordType>(ty).getFieldIndex(fieldName);
        indices.push_back(index);
      }
    }
    return indices;
  }

private:
  static aiir::Type getArrayElementType(aiir::LLVM::LLVMArrayType ty) {
    auto eleTy = ty.getElementType();
    while (auto arrTy = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(eleTy))
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
  doRewrite(fir::ExtractValueOp extractVal, aiir::Type ty, OpAdaptor adaptor,
            aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();
    auto indices = collectIndices(rewriter, extractVal.getCoor());
    toRowMajor(indices, operands[0].getType());
    rewriter.replaceOpWithNewOp<aiir::LLVM::ExtractValueOp>(
        extractVal, operands[0], indices);
    return aiir::success();
  }
};

/// InsertValue is the generalized instruction for the composition of new
/// aggregate type values.
struct InsertValueOpConversion
    : public aiir::OpConversionPattern<fir::InsertValueOp>,
      public ValueOpCommon {
  using OpConversionPattern::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(fir::InsertValueOp insertVal, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();
    auto indices = collectIndices(rewriter, insertVal.getCoor());
    toRowMajor(indices, operands[0].getType());
    rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(
        insertVal, operands[0], operands[1], indices);
    return aiir::success();
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
  doRewrite(fir::InsertOnRangeOp range, aiir::Type ty, OpAdaptor adaptor,
            aiir::ConversionPatternRewriter &rewriter) const override {

    auto arrayType = adaptor.getSeq().getType();

    // Iteratively extract the array dimensions from the type.
    llvm::SmallVector<std::int64_t> dims;
    aiir::Type type = arrayType;
    while (auto t = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(type)) {
      dims.push_back(t.getNumElements());
      type = t.getElementType();
    }

    // Avoid generating long insert chain that are very slow to fold back
    // (which is required in globals when later generating LLVM IR). Attempt to
    // fold the inserted element value to an attribute and build an ArrayAttr
    // for the resulting array.
    if (range.isFullRange()) {
      llvm::FailureOr<aiir::Attribute> cst =
          fir::tryFoldingLLVMInsertChain(adaptor.getVal(), rewriter);
      if (llvm::succeeded(cst)) {
        aiir::Attribute dimVal = *cst;
        for (auto dim : llvm::reverse(dims)) {
          // Use std::vector in case the number of elements is big.
          std::vector<aiir::Attribute> elements(dim, dimVal);
          dimVal = aiir::ArrayAttr::get(range.getContext(), elements);
        }
        // Replace insert chain with constant.
        rewriter.replaceOpWithNewOp<aiir::LLVM::ConstantOp>(range, arrayType,
                                                            dimVal);
        return aiir::success();
      }
    }

    // The inserted value cannot be folded to an attribute, turn the
    // insert_range into an llvm.insertvalue chain.
    llvm::SmallVector<std::int64_t> lBounds;
    llvm::SmallVector<std::int64_t> uBounds;

    // Unzip the upper and lower bound and convert to a row major format.
    aiir::DenseIntElementsAttr coor = range.getCoor();
    auto reversedCoor = llvm::reverse(coor.getValues<int64_t>());
    for (auto i = reversedCoor.begin(), e = reversedCoor.end(); i != e; ++i) {
      uBounds.push_back(*i++);
      lBounds.push_back(*i);
    }

    auto &subscripts = lBounds;
    auto loc = range.getLoc();
    aiir::Value lastOp = adaptor.getSeq();
    aiir::Value insertVal = adaptor.getVal();

    while (subscripts != uBounds) {
      lastOp = aiir::LLVM::InsertValueOp::create(rewriter, loc, lastOp,
                                                 insertVal, subscripts);

      incrementSubscripts(dims, subscripts);
    }

    rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(
        range, lastOp, insertVal, subscripts);

    return aiir::success();
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
  doRewrite(fir::cg::XArrayCoorOp coor, aiir::Type llvmPtrTy, OpAdaptor adaptor,
            aiir::ConversionPatternRewriter &rewriter) const override {
    auto loc = coor.getLoc();
    aiir::ValueRange operands = adaptor.getOperands();
    unsigned rank = coor.getRank();
    assert(coor.getIndices().size() == rank);
    assert(coor.getShape().empty() || coor.getShape().size() == rank);
    assert(coor.getShift().empty() || coor.getShift().size() == rank);
    assert(coor.getSlice().empty() || coor.getSlice().size() == 3 * rank);
    aiir::Type idxTy = lowerTy().indexType();
    unsigned indexOffset = coor.getIndicesOperandIndex();
    unsigned shapeOffset = coor.getShapeOperandIndex();
    unsigned shiftOffset = coor.getShiftOperandIndex();
    unsigned sliceOffset = coor.getSliceOperandIndex();
    auto sliceOps = coor.getSlice().begin();
    aiir::Value one = fir::genConstantIndex(loc, idxTy, rewriter, 1);
    aiir::Value prevExt = one;
    aiir::Value offset = fir::genConstantIndex(loc, idxTy, rewriter, 0);
    const bool isShifted = !coor.getShift().empty();
    const bool isSliced = !coor.getSlice().empty();
    const bool baseIsBoxed =
        aiir::isa<fir::BaseBoxType>(coor.getMemref().getType());
    TypePair baseBoxTyPair =
        baseIsBoxed ? getBoxTypePair(coor.getMemref().getType()) : TypePair{};
    aiir::LLVM::IntegerOverflowFlags nsw =
        aiir::LLVM::IntegerOverflowFlags::nsw;
    aiir::LLVM::IntegerOverflowFlags nuw =
        aiir::LLVM::IntegerOverflowFlags::nuw;
    // TODO Allow for non-default lower bounds that are positive
    // We know at compile time this is possible, so could be updated in future
    // to allow for this, and just exclude non-default lower bounds that are
    // negative. Currently, all shifted XArrayCoorOp's only have nsw on sub
    // operations.
    aiir::LLVM::IntegerOverflowFlags subFlags = isShifted ? nsw : (nsw | nuw);
    aiir::LLVM::IntegerOverflowFlags addMulFlags = nsw | nuw;
    aiir::LLVM::GEPNoWrapFlags gepFlags = aiir::LLVM::GEPNoWrapFlags::nusw | aiir::LLVM::GEPNoWrapFlags::nuw;

    // For each dimension of the array, generate the offset calculation.
    for (unsigned i = 0; i < rank; ++i, ++indexOffset, ++shapeOffset,
                  ++shiftOffset, sliceOffset += 3, sliceOps += 3) {
      aiir::Value index =
          integerCast(loc, rewriter, idxTy, operands[indexOffset]);
      aiir::Value lb =
          isShifted ? integerCast(loc, rewriter, idxTy, operands[shiftOffset])
                    : one;
      aiir::Value step = one;
      bool normalSlice = isSliced;
      // Compute zero based index in dimension i of the element, applying
      // potential triplets and lower bounds.
      if (isSliced) {
        aiir::Value originalUb = *(sliceOps + 1);
        normalSlice =
            !aiir::isa_and_nonnull<fir::UndefOp>(originalUb.getDefiningOp());
        if (normalSlice)
          step = integerCast(loc, rewriter, idxTy, operands[sliceOffset + 2]);
      }
      auto idx =
          aiir::LLVM::SubOp::create(rewriter, loc, idxTy, index, lb, subFlags);
      aiir::Value diff = aiir::LLVM::MulOp::create(rewriter, loc, idxTy, idx,
                                                   step, addMulFlags);
      if (normalSlice) {
        aiir::Value sliceLb =
            integerCast(loc, rewriter, idxTy, operands[sliceOffset]);
        auto adj = aiir::LLVM::SubOp::create(rewriter, loc, idxTy, sliceLb, lb,
                                             subFlags);
        diff = aiir::LLVM::AddOp::create(rewriter, loc, idxTy, diff, adj,
                                         addMulFlags);
      }
      // Update the offset given the stride and the zero based index `diff`
      // that was just computed.
      if (baseIsBoxed) {
        // Use stride in bytes from the descriptor.
        aiir::Value stride =
            getStrideFromBox(loc, baseBoxTyPair, operands[0], i, rewriter);
        auto sc = aiir::LLVM::MulOp::create(rewriter, loc, idxTy, diff, stride,
                                            addMulFlags);
        offset = aiir::LLVM::AddOp::create(rewriter, loc, idxTy, sc, offset,
                                           addMulFlags);
      } else {
        // Use stride computed at last iteration.
        auto sc = aiir::LLVM::MulOp::create(rewriter, loc, idxTy, diff, prevExt,
                                            addMulFlags);
        offset = aiir::LLVM::AddOp::create(rewriter, loc, idxTy, sc, offset,
                                           addMulFlags);
        // Compute next stride assuming contiguity of the base array
        // (in element number).
        auto nextExt = integerCast(loc, rewriter, idxTy, operands[shapeOffset]);
        prevExt = aiir::LLVM::MulOp::create(rewriter, loc, idxTy, prevExt,
                                            nextExt, addMulFlags);
      }
    }

    // Add computed offset to the base address.
    if (baseIsBoxed) {
      // Working with byte offsets. The base address is read from the fir.box.
      // and used in i8* GEP to do the pointer arithmetic.
      aiir::Type byteTy = ::getI8Type(coor.getContext());
      aiir::Value base =
          getBaseAddrFromBox(loc, baseBoxTyPair, operands[0], rewriter);
      llvm::SmallVector<aiir::LLVM::GEPArg> args{offset};
      auto addr = aiir::LLVM::GEPOp::create(rewriter, loc, llvmPtrTy, byteTy,
                                            base, args, gepFlags);
      if (coor.getSubcomponent().empty()) {
        rewriter.replaceOp(coor, addr);
        return aiir::success();
      }
      // Cast the element address from void* to the derived type so that the
      // derived type members can be addresses via a GEP using the index of
      // components.
      aiir::Type elementType =
          getLlvmObjectTypeFromBoxType(coor.getMemref().getType());
      while (auto arrayTy =
                 aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(elementType))
        elementType = arrayTy.getElementType();
      args.clear();
      args.push_back(0);
      if (!coor.getLenParams().empty()) {
        // If type parameters are present, then we don't want to use a GEPOp
        // as below, as the LLVM struct type cannot be statically defined.
        TODO(loc, "derived type with type parameters");
      }
      llvm::SmallVector<aiir::Value> indices = convertSubcomponentIndices(
          loc, elementType,
          operands.slice(coor.getSubcomponentOperandIndex(),
                         coor.getSubcomponent().size()));
      args.append(indices.begin(), indices.end());
      rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
          coor, llvmPtrTy, elementType, addr, args, gepFlags);
      return aiir::success();
    }

    // The array was not boxed, so it must be contiguous. offset is therefore an
    // element offset and the base type is kept in the GEP unless the element
    // type size is itself dynamic.
    aiir::Type objectTy = fir::unwrapRefType(coor.getMemref().getType());
    aiir::Type eleType = fir::unwrapSequenceType(objectTy);
    aiir::Type gepObjectType = convertType(eleType);
    llvm::SmallVector<aiir::LLVM::GEPArg> args;
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
          offset = aiir::LLVM::MulOp::create(rewriter, loc, idxTy, offset,
                                             length, addMulFlags);
        } else {
          TODO(loc, "compute size of derived type with type parameters");
        }
      }
      args.push_back(offset);
    } else {
      // There are subcomponents.
      args.push_back(offset);
      llvm::SmallVector<aiir::Value> indices = convertSubcomponentIndices(
          loc, gepObjectType,
          operands.slice(coor.getSubcomponentOperandIndex(),
                         coor.getSubcomponent().size()));
      args.append(indices.begin(), indices.end());
    }
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
        coor, llvmPtrTy, gepObjectType, adaptor.getMemref(), args, gepFlags);
    return aiir::success();
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
  doRewrite(fir::CoordinateOp coor, aiir::Type ty, OpAdaptor adaptor,
            aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::ValueRange operands = adaptor.getOperands();

    aiir::Location loc = coor.getLoc();
    aiir::Value base = operands[0];
    aiir::Type baseObjectTy = coor.getBaseType();
    aiir::Type objectTy = fir::dyn_cast_ptrOrBoxEleTy(baseObjectTy);
    assert(objectTy && "fir.coordinate_of expects a reference type");
    aiir::Type llvmObjectTy = convertType(objectTy);

    // Complex type - basically, extract the real or imaginary part
    // FIXME: double check why this is done before the fir.box case below.
    if (fir::isa_complex(objectTy)) {
      aiir::Value gep =
          genGEP(loc, llvmObjectTy, rewriter, base, 0, operands[1]);
      rewriter.replaceOp(coor, gep);
      return aiir::success();
    }

    // Boxed type - get the base pointer from the box
    if (aiir::dyn_cast<fir::BaseBoxType>(baseObjectTy))
      return doRewriteBox(coor, operands, loc, rewriter);

    // Reference, pointer or a heap type
    if (aiir::isa<fir::ReferenceType, fir::PointerType, fir::HeapType>(
            baseObjectTy))
      return doRewriteRefOrPtr(coor, llvmObjectTy, operands, loc, rewriter);

    return rewriter.notifyMatchFailure(
        coor, "fir.coordinate_of base operand has unsupported type");
  }

  static unsigned getFieldNumber(fir::RecordType ty, aiir::Value op) {
    return fir::hasDynamicSize(ty)
               ? op.getDefiningOp()
                     ->getAttrOfType<aiir::IntegerAttr>("field")
                     .getInt()
               : getConstantIntValue(op);
  }

  static bool hasSubDimensions(aiir::Type type) {
    return aiir::isa<fir::SequenceType, fir::RecordType, aiir::TupleType>(type);
  }

  // Helper structure to analyze the CoordinateOp path and decide if and how
  // the GEP should be generated for it.
  struct ShapeAnalysis {
    bool hasKnownShape;
    bool columnIsDeferred;
  };

  /// Walk the abstract memory layout and determine if the path traverses any
  /// array types with unknown shape. Return true iff all the array types have a
  /// constant shape along the path.
  /// TODO: move the verification logic into the verifier.
  static std::optional<ShapeAnalysis>
  arraysHaveKnownShape(aiir::Type type, fir::CoordinateOp coor) {
    fir::CoordinateIndicesAdaptor indices = coor.getIndices();
    auto begin = indices.begin();
    bool hasKnownShape = true;
    bool columnIsDeferred = false;
    for (auto it = begin, end = indices.end(); it != end;) {
      if (auto arrTy = aiir::dyn_cast<fir::SequenceType>(type)) {
        bool addressingStart = (it == begin);
        unsigned arrayDim = arrTy.getDimension();
        for (auto dimExtent : llvm::enumerate(arrTy.getShape())) {
          if (dimExtent.value() == fir::SequenceType::getUnknownExtent()) {
            hasKnownShape = false;
            if (addressingStart && dimExtent.index() + 1 == arrayDim) {
              // If this point was reached, the raws of the first array have
              // constant extents.
              columnIsDeferred = true;
            } else {
              // One of the array dimension that is not the column of the first
              // array has dynamic extent. It will not possible to do
              // code generation for the CoordinateOp if the base is not a
              // fir.box containing the value of that extent.
              return ShapeAnalysis{false, false};
            }
          }
          // There may be less operands than the array size if the
          // fir.coordinate_of result is not an element but a sub-array.
          if (it != end)
            ++it;
        }
        type = arrTy.getEleTy();
        continue;
      }
      if (auto strTy = aiir::dyn_cast<fir::RecordType>(type)) {
        auto intAttr = llvm::dyn_cast<aiir::IntegerAttr>(*it);
        if (!intAttr) {
          aiir::emitError(coor.getLoc(),
                          "expected field name in fir.coordinate_of");
          return std::nullopt;
        }
        type = strTy.getType(intAttr.getInt());
      } else if (auto strTy = aiir::dyn_cast<aiir::TupleType>(type)) {
        auto value = llvm::dyn_cast<aiir::Value>(*it);
        if (!value) {
          aiir::emitError(
              coor.getLoc(),
              "expected constant value to address tuple in fir.coordinate_of");
          return std::nullopt;
        }
        type = strTy.getType(getConstantIntValue(value));
      } else if (auto charType = aiir::dyn_cast<fir::CharacterType>(type)) {
        // Addressing character in string. Fortran strings degenerate to arrays
        // in LLVM, so they are handled like arrays of characters here.
        if (charType.getLen() == fir::CharacterType::unknownLen())
          return ShapeAnalysis{false, true};
        type = fir::CharacterType::getSingleton(charType.getContext(),
                                                charType.getFKind());
      }
      ++it;
    }
    return ShapeAnalysis{hasKnownShape, columnIsDeferred};
  }

private:
  llvm::LogicalResult
  doRewriteBox(fir::CoordinateOp coor, aiir::ValueRange operands,
               aiir::Location loc,
               aiir::ConversionPatternRewriter &rewriter) const {
    aiir::Type boxObjTy = coor.getBaseType();
    assert(aiir::dyn_cast<fir::BaseBoxType>(boxObjTy) &&
           "This is not a `fir.box`");
    TypePair boxTyPair = getBoxTypePair(boxObjTy);

    aiir::Value boxBaseAddr = operands[0];

    // 1. SPECIAL CASE (uses `fir.len_param_index`):
    //   %box = ... : !fir.box<!fir.type<derived{len1:i32}>>
    //   %lenp = fir.len_param_index len1, !fir.type<derived{len1:i32}>
    //   %addr = coordinate_of %box, %lenp
    if (coor.getNumOperands() == 2) {
      aiir::Operation *coordinateDef =
          (*coor.getCoor().begin()).getDefiningOp();
      if (aiir::isa_and_nonnull<fir::LenParamIndexOp>(coordinateDef))
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
    aiir::Value resultAddr =
        getBaseAddrFromBox(loc, boxTyPair, boxBaseAddr, rewriter);
    // Component Type
    auto cpnTy = fir::dyn_cast_ptrOrBoxEleTy(boxObjTy);
    aiir::Type llvmPtrTy = ::getLlvmPtrType(coor.getContext());
    aiir::Type byteTy = ::getI8Type(coor.getContext());
    aiir::LLVM::IntegerOverflowFlags nsw =
        aiir::LLVM::IntegerOverflowFlags::nsw;

    int nextIndexValue = 1;
    fir::CoordinateIndicesAdaptor indices = coor.getIndices();
    for (auto it = indices.begin(), end = indices.end(); it != end;) {
      if (auto arrTy = aiir::dyn_cast<fir::SequenceType>(cpnTy)) {
        if (it != indices.begin())
          TODO(loc, "fir.array nested inside other array and/or derived type");
        // Applies byte strides from the box. Ignore lower bound from box
        // since fir.coordinate_of indexes are zero based. Lowering takes care
        // of lower bound aspects. This both accounts for dynamically sized
        // types and non contiguous arrays.
        auto idxTy = lowerTy().indexType();
        aiir::Value off = fir::genConstantIndex(loc, idxTy, rewriter, 0);
        unsigned arrayDim = arrTy.getDimension();
        for (unsigned dim = 0; dim < arrayDim && it != end; ++dim, ++it) {
          aiir::Value stride =
              getStrideFromBox(loc, boxTyPair, operands[0], dim, rewriter);
          auto sc = aiir::LLVM::MulOp::create(rewriter, loc, idxTy,
                                              operands[nextIndexValue + dim],
                                              stride, nsw);
          off = aiir::LLVM::AddOp::create(rewriter, loc, idxTy, sc, off, nsw);
        }
        nextIndexValue += arrayDim;
        resultAddr = aiir::LLVM::GEPOp::create(
            rewriter, loc, llvmPtrTy, byteTy, resultAddr,
            llvm::ArrayRef<aiir::LLVM::GEPArg>{off});
        cpnTy = arrTy.getEleTy();
      } else if (auto recTy = aiir::dyn_cast<fir::RecordType>(cpnTy)) {
        auto intAttr = llvm::dyn_cast<aiir::IntegerAttr>(*it);
        if (!intAttr)
          return aiir::emitError(loc,
                                 "expected field name in fir.coordinate_of");
        int fieldIndex = intAttr.getInt();
        ++it;
        cpnTy = recTy.getType(fieldIndex);
        auto llvmRecTy = lowerTy().convertType(recTy);
        resultAddr = aiir::LLVM::GEPOp::create(
            rewriter, loc, llvmPtrTy, llvmRecTy, resultAddr,
            llvm::ArrayRef<aiir::LLVM::GEPArg>{0, fieldIndex});
      } else {
        fir::emitFatalError(loc, "unexpected type in coordinate_of");
      }
    }

    rewriter.replaceOp(coor, resultAddr);
    return aiir::success();
  }

  llvm::LogicalResult
  doRewriteRefOrPtr(fir::CoordinateOp coor, aiir::Type llvmObjectTy,
                    aiir::ValueRange operands, aiir::Location loc,
                    aiir::ConversionPatternRewriter &rewriter) const {
    aiir::Type baseObjectTy = coor.getBaseType();

    // Component Type
    aiir::Type cpnTy = fir::dyn_cast_ptrOrBoxEleTy(baseObjectTy);

    const std::optional<ShapeAnalysis> shapeAnalysis =
        arraysHaveKnownShape(cpnTy, coor);
    if (!shapeAnalysis)
      return aiir::failure();

    if (fir::hasDynamicSize(fir::unwrapSequenceType(cpnTy)))
      return aiir::emitError(
          loc, "fir.coordinate_of with a dynamic element size is unsupported");

    if (shapeAnalysis->hasKnownShape || shapeAnalysis->columnIsDeferred) {
      llvm::SmallVector<aiir::LLVM::GEPArg> offs;
      if (shapeAnalysis->hasKnownShape) {
        offs.push_back(0);
      }
      // Else, only the column is `?` and we can simply place the column value
      // in the 0-th GEP position.

      std::optional<int> dims;
      llvm::SmallVector<aiir::Value> arrIdx;
      int nextIndexValue = 1;
      for (auto index : coor.getIndices()) {
        if (auto intAttr = llvm::dyn_cast<aiir::IntegerAttr>(index)) {
          // Addressing derived type component.
          auto recordType = llvm::dyn_cast<fir::RecordType>(cpnTy);
          if (!recordType)
            return aiir::emitError(
                loc,
                "fir.coordinate base type is not consistent with operands");
          int fieldId = intAttr.getInt();
          cpnTy = recordType.getType(fieldId);
          offs.push_back(fieldId);
          continue;
        }
        // Value index (addressing array, tuple, or complex part).
        aiir::Value indexValue = operands[nextIndexValue++];
        if (auto tupTy = aiir::dyn_cast<aiir::TupleType>(cpnTy)) {
          cpnTy = tupTy.getType(getConstantIntValue(indexValue));
          offs.push_back(indexValue);
        } else {
          if (!dims) {
            if (auto arrayType = llvm::dyn_cast<fir::SequenceType>(cpnTy)) {
              // Starting addressing array or array component.
              dims = arrayType.getDimension();
              cpnTy = arrayType.getElementType();
            }
          }
          if (dims) {
            arrIdx.push_back(indexValue);
            if (--(*dims) == 0) {
              // Append array range in reverse (FIR arrays are column-major).
              offs.append(arrIdx.rbegin(), arrIdx.rend());
              arrIdx.clear();
              dims.reset();
            }
          } else {
            offs.push_back(indexValue);
          }
        }
      }
      // It is possible the fir.coordinate_of result is a sub-array, in which
      // case there may be some "unfinished" array indices to reverse and push.
      if (!arrIdx.empty())
        offs.append(arrIdx.rbegin(), arrIdx.rend());

      aiir::Value base = operands[0];
      aiir::Value retval = genGEP(loc, llvmObjectTy, rewriter, base, offs);
      rewriter.replaceOp(coor, retval);
      return aiir::success();
    }

    return aiir::emitError(
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    auto recTy = aiir::cast<fir::RecordType>(field.getOnType());
    unsigned index = recTy.getFieldIndex(field.getFieldId());

    if (!fir::hasDynamicSize(recTy)) {
      // Derived type has compile-time constant layout. Return index of the
      // component type in the parent type (to be used in GEP).
      rewriter.replaceOp(field, aiir::ValueRange{genConstantOffset(
                                    field.getLoc(), rewriter, index)});
      return aiir::success();
    }

    // Derived type has compile-time constant layout. Call the compiler
    // generated function to determine the byte offset of the field at runtime.
    // This returns a non-constant.
    aiir::FlatSymbolRefAttr symAttr = aiir::SymbolRefAttr::get(
        field.getContext(), getOffsetMethodName(recTy, field.getFieldId()));
    aiir::NamedAttribute callAttr = rewriter.getNamedAttr("callee", symAttr);
    aiir::NamedAttribute fieldAttr = rewriter.getNamedAttr(
        "field", aiir::IntegerAttr::get(lowerTy().indexType(), index));
    rewriter.replaceOpWithNewOp<aiir::LLVM::CallOp>(
        field, lowerTy().offsetType(), adaptor.getOperands(),
        addLLVMOpBundleAttrs(rewriter, {callAttr, fieldAttr},
                             adaptor.getOperands().size()));
    return aiir::success();
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    TODO(firEnd.getLoc(), "fir.end codegen");
    return aiir::failure();
  }
};

/// Lower `fir.type_desc` to a global addr.
struct TypeDescOpConversion : public fir::FIROpConversion<fir::TypeDescOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::TypeDescOp typeDescOp, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type inTy = typeDescOp.getInType();
    assert(aiir::isa<fir::RecordType>(inTy) && "expecting fir.type");
    auto recordType = aiir::dyn_cast<fir::RecordType>(inTy);
    auto module = typeDescOp.getOperation()->getParentOfType<aiir::ModuleOp>();
    aiir::Value typeDesc = getTypeDescriptor(
        module, rewriter, typeDescOp.getLoc(), recordType, this->options);
    rewriter.replaceOp(typeDescOp, typeDesc);
    return aiir::success();
  }
};

/// Lower `fir.has_value` operation to `llvm.return` operation.
struct HasValueOpConversion
    : public aiir::OpConversionPattern<fir::HasValueOp> {
  using OpConversionPattern::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(fir::HasValueOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<aiir::LLVM::ReturnOp>(op,
                                                      adaptor.getOperands());
    return aiir::success();
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
static inline bool attributeTypeIsCompatible(aiir::AIIRContext *ctx,
                                             aiir::Attribute attr,
                                             aiir::Type ty) {
  // Get attr's LLVM element type.
  if (!attr)
    return true;
  auto intOrFpEleAttr = aiir::dyn_cast<aiir::DenseTypedElementsAttr>(attr);
  if (!intOrFpEleAttr)
    return true;
  auto tensorTy = aiir::dyn_cast<aiir::TensorType>(intOrFpEleAttr.getType());
  if (!tensorTy)
    return true;
  aiir::Type attrEleTy =
      aiir::LLVMTypeConverter(ctx).convertType(tensorTy.getElementType());

  // Get ty's element type.
  auto arrTy = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(ty);
  if (!arrTy)
    return true;
  aiir::Type eleTy = arrTy.getElementType();
  while ((arrTy = aiir::dyn_cast<aiir::LLVM::LLVMArrayType>(eleTy)))
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
                  aiir::ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<aiir::Attribute> dbgExprs;

    if (auto fusedLoc = aiir::dyn_cast<aiir::FusedLoc>(global.getLoc())) {
      if (auto gvExprAttr = aiir::dyn_cast_if_present<aiir::ArrayAttr>(
              fusedLoc.getMetadata())) {
        for (auto attr : gvExprAttr.getAsRange<aiir::Attribute>())
          if (auto dbgAttr =
                  aiir::dyn_cast<aiir::LLVM::DIGlobalVariableExpressionAttr>(
                      attr))
            dbgExprs.push_back(dbgAttr);
      }
    }

    auto tyAttr = convertType(global.getType());
    if (auto boxType = aiir::dyn_cast<fir::BaseBoxType>(global.getType()))
      tyAttr = this->lowerTy().convertBoxTypeAsStruct(boxType);
    auto loc = global.getLoc();
    aiir::Attribute initAttr = global.getInitVal().value_or(aiir::Attribute());
    assert(attributeTypeIsCompatible(global.getContext(), initAttr, tyAttr));
    auto linkage = convertLinkage(global.getLinkName());
    auto isConst = global.getConstant().has_value();
    aiir::SymbolRefAttr comdat;
    llvm::ArrayRef<aiir::NamedAttribute> attrs;
    auto g = aiir::LLVM::GlobalOp::create(
        rewriter, loc, tyAttr, isConst, linkage, global.getSymName(), initAttr,
        0, getGlobalAddressSpace(rewriter), false, false, comdat, attrs,
        dbgExprs);

    if (global.getAlignment() && *global.getAlignment() > 0)
      g.setAlignment(*global.getAlignment());

    auto module = global->getParentOfType<aiir::ModuleOp>();
    auto gpuMod = global->getParentOfType<aiir::gpu::GPUModuleOp>();
    // Add comdat if necessary
    if (fir::getTargetTriple(module).supportsCOMDAT() &&
        (linkage == aiir::LLVM::Linkage::Linkonce ||
         linkage == aiir::LLVM::Linkage::LinkonceODR) &&
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
        if (insertOp.isFullRange()) {
          auto seqTyAttr = convertType(insertOp.getType());
          auto *op = insertOp.getVal().getDefiningOp();
          auto constant = aiir::dyn_cast<aiir::arith::ConstantOp>(op);
          if (!constant) {
            auto convertOp = aiir::dyn_cast<fir::ConvertOp>(op);
            if (!convertOp)
              continue;
            constant = aiir::cast<aiir::arith::ConstantOp>(
                convertOp.getValue().getDefiningOp());
          }
          aiir::Type vecType = aiir::VectorType::get(
              insertOp.getType().getShape(), constant.getType());
          auto denseAttr = aiir::DenseElementsAttr::get(
              aiir::cast<aiir::ShapedType>(vecType), constant.getValue());
          rewriter.setInsertionPointAfter(insertOp);
          rewriter.replaceOpWithNewOp<aiir::arith::ConstantOp>(
              insertOp, seqTyAttr, denseAttr);
        }
      }
    }

    if (global.getDataAttr() &&
        *global.getDataAttr() == cuf::DataAttribute::Shared)
      g.setAddrSpace(
          static_cast<unsigned>(aiir::NVVM::NVVMMemorySpace::Shared));

    if (global.getDataAttr() &&
        *global.getDataAttr() == cuf::DataAttribute::Constant)
      g.setAddrSpace(
          static_cast<unsigned>(aiir::NVVM::NVVMMemorySpace::Constant));

    rewriter.eraseOp(global);
    return aiir::success();
  }

  // TODO: String comparisons should be avoided. Replace linkName with an
  // enumeration.
  aiir::LLVM::Linkage
  convertLinkage(std::optional<llvm::StringRef> optLinkage) const {
    if (optLinkage) {
      auto name = *optLinkage;
      if (name == "internal")
        return aiir::LLVM::Linkage::Internal;
      if (name == "linkonce")
        return aiir::LLVM::Linkage::Linkonce;
      if (name == "linkonce_odr")
        return aiir::LLVM::Linkage::LinkonceODR;
      if (name == "common")
        return aiir::LLVM::Linkage::Common;
      if (name == "weak")
        return aiir::LLVM::Linkage::Weak;
    }
    return aiir::LLVM::Linkage::External;
  }

private:
  static void addComdat(aiir::LLVM::GlobalOp &global,
                        aiir::ConversionPatternRewriter &rewriter,
                        aiir::ModuleOp module) {
    const char *comdatName = "__llvm_comdat";
    aiir::LLVM::ComdatOp comdatOp =
        module.lookupSymbol<aiir::LLVM::ComdatOp>(comdatName);
    if (!comdatOp) {
      comdatOp =
          aiir::LLVM::ComdatOp::create(rewriter, module.getLoc(), comdatName);
    }
    if (auto select = comdatOp.lookupSymbol<aiir::LLVM::ComdatSelectorOp>(
            global.getSymName()))
      return;
    aiir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&comdatOp.getBody().back());
    auto selectorOp = aiir::LLVM::ComdatSelectorOp::create(
        rewriter, comdatOp.getLoc(), global.getSymName(),
        aiir::LLVM::comdat::Comdat::Any);
    global.setComdatAttr(aiir::SymbolRefAttr::get(
        rewriter.getContext(), comdatName,
        aiir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
  }
};

/// `fir.prefetch` --> `llvm.prefetch`
struct PrefetchOpConversion : public fir::FIROpConversion<fir::PrefetchOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::PrefetchOp prefetch, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::IntegerAttr rw = aiir::IntegerAttr::get(rewriter.getI32Type(),
                                                  prefetch.getRwAttr() ? 1 : 0);
    aiir::IntegerAttr localityHint = prefetch.getLocalityHintAttr();
    aiir::IntegerAttr cacheType = aiir::IntegerAttr::get(
        rewriter.getI32Type(), prefetch.getCacheTypeAttr() ? 1 : 0);
    aiir::LLVM::Prefetch::create(rewriter, prefetch.getLoc(),
                                 adaptor.getOperands().front(), rw,
                                 localityHint, cacheType);
    rewriter.eraseOp(prefetch);
    return aiir::success();
  }
};

/// `fir.load` --> `llvm.load`
struct LoadOpConversion : public fir::FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::LoadOp load, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {

    aiir::Type llvmLoadTy = convertObjectType(load.getType());
    const bool isVolatile = fir::isa_volatile_type(load.getMemref().getType());
    if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(load.getType())) {
      // fir.box is a special case because it is considered an ssa value in
      // fir, but it is lowered as a pointer to a descriptor. So
      // fir.ref<fir.box> and fir.box end up being the same llvm types and
      // loading a fir.ref<fir.box> is implemented as taking a snapshot of the
      // descriptor value into a new descriptor temp.
      auto inputBoxStorage = adaptor.getOperands()[0];
      aiir::Value newBoxStorage;
      aiir::Location loc = load.getLoc();
      if (auto callOp = aiir::dyn_cast_or_null<aiir::LLVM::CallOp>(
              inputBoxStorage.getDefiningOp())) {
        if (callOp.getCallee() &&
            ((*callOp.getCallee())
                 .starts_with(RTNAME_STRING(CUFAllocDescriptor)) ||
             (*callOp.getCallee()).starts_with("__tgt_acc_get_deviceptr"))) {
          // CUDA Fortran local descriptor are allocated in managed memory. So
          // new storage must be allocated the same way.
          auto mod = load->getParentOfType<aiir::ModuleOp>();
          newBoxStorage =
              genCUFAllocDescriptor(loc, rewriter, mod, boxTy, lowerTy());
        }
      }
      if (!newBoxStorage)
        newBoxStorage = genAllocaAndAddrCastWithType(loc, llvmLoadTy,
                                                     defaultAlign, rewriter);

      TypePair boxTypePair{boxTy, llvmLoadTy};
      aiir::Value boxSize =
          computeBoxSize(loc, boxTypePair, inputBoxStorage, rewriter);
      auto memcpy = aiir::LLVM::MemcpyOp::create(
          rewriter, loc, newBoxStorage, inputBoxStorage, boxSize, isVolatile);
      setMemcpyAlignmentArgAttrs(memcpy, rewriter, getDataLayout(), llvmLoadTy);

      if (std::optional<aiir::ArrayAttr> optionalTag = load.getTbaa())
        memcpy.setTBAATags(*optionalTag);
      else
        attachTBAATag(memcpy, boxTy, boxTy, nullptr);

      rewriter.replaceOp(load, newBoxStorage);
    } else {
      aiir::LLVM::LoadOp loadOp =
          aiir::LLVM::LoadOp::create(rewriter, load.getLoc(), llvmLoadTy,
                                     adaptor.getOperands(), load->getAttrs());
      loadOp.setVolatile_(isVolatile);
      if (std::optional<aiir::ArrayAttr> optionalTag = load.getTbaa())
        loadOp.setTBAATags(*optionalTag);
      else
        attachTBAATag(loadOp, load.getType(), load.getType(), nullptr);
      if (std::optional<aiir::ArrayAttr> optionalAccessGroups =
              load.getAccessGroups())
        loadOp.setAccessGroups(*optionalAccessGroups);
      rewriter.replaceOp(load, loadOp.getResult());
    }
    return aiir::success();
  }
};

template <typename OpTy>
struct DoConcurrentSpecifierOpConversion : public fir::FIROpConversion<OpTy> {
  using fir::FIROpConversion<OpTy>::FIROpConversion;
  llvm::LogicalResult
  matchAndRewrite(OpTy specifier, typename OpTy::Adaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
#ifdef EXPENSIVE_CHECKS
    auto uses = aiir::SymbolTable::getSymbolUses(
        specifier, specifier->template getParentOfType<aiir::ModuleOp>());

    // `fir.local|fir.declare_reduction` ops are not supposed to have any uses
    // at this point (i.e. during lowering to LLVM). In case of serialization,
    // the `fir.do_concurrent` users are expected to have been lowered to
    // `fir.do_loop` nests. In case of parallelization, the `fir.do_concurrent`
    // users are expected to have been lowered to the target parallel model
    // (e.g. OpenMP).
    assert(uses && uses->empty());
#endif

    rewriter.eraseOp(specifier);
    return aiir::success();
  }
};

/// Lower `fir.no_reassoc` to LLVM IR dialect.
/// TODO: how do we want to enforce this in LLVM-IR? Can we manipulate the fast
/// math flags?
struct NoReassocOpConversion : public fir::FIROpConversion<fir::NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::NoReassocOp noreassoc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(noreassoc, adaptor.getOperands()[0]);
    return aiir::success();
  }
};

/// Erase `fir.use_stmt` operations during LLVM lowering.
/// These operations are only used for debug info generation by the
/// AddDebugInfo pass and have no runtime representation.
struct UseStmtOpConversion : public fir::FIROpConversion<fir::UseStmtOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UseStmtOp useStmt, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(useStmt);
    return aiir::success();
  }
};

static void genCondBrOp(aiir::Location loc, aiir::Value cmp, aiir::Block *dest,
                        std::optional<aiir::ValueRange> destOps,
                        aiir::ConversionPatternRewriter &rewriter,
                        aiir::Block *newBlock) {
  if (destOps)
    aiir::LLVM::CondBrOp::create(rewriter, loc, cmp, dest, *destOps, newBlock,
                                 aiir::ValueRange());
  else
    aiir::LLVM::CondBrOp::create(rewriter, loc, cmp, dest, newBlock);
}

template <typename A, typename B>
static void genBrOp(A caseOp, aiir::Block *dest, std::optional<B> destOps,
                    aiir::ConversionPatternRewriter &rewriter) {
  if (destOps)
    rewriter.replaceOpWithNewOp<aiir::LLVM::BrOp>(caseOp, *destOps, dest);
  else
    rewriter.replaceOpWithNewOp<aiir::LLVM::BrOp>(caseOp, B{}, dest);
}

static void genCaseLadderStep(aiir::Location loc, aiir::Value cmp,
                              aiir::Block *dest,
                              std::optional<aiir::ValueRange> destOps,
                              aiir::ConversionPatternRewriter &rewriter) {
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
                  aiir::ConversionPatternRewriter &rewriter) const override {
    unsigned conds = caseOp.getNumConditions();
    llvm::ArrayRef<aiir::Attribute> cases = caseOp.getCases().getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    auto ty = caseOp.getSelector().getType();
    if (aiir::isa<fir::CharacterType>(ty)) {
      TODO(caseOp.getLoc(), "fir.select_case codegen with character type");
      return aiir::failure();
    }
    aiir::Value selector = caseOp.getSelector(adaptor.getOperands());
    auto loc = caseOp.getLoc();
    for (unsigned t = 0; t != conds; ++t) {
      aiir::Block *dest = caseOp.getSuccessor(t);
      std::optional<aiir::ValueRange> destOps =
          caseOp.getSuccessorOperands(adaptor.getOperands(), t);
      // Convert block signature if needed
      if (destOps && !destOps->empty())
        if (auto conversion = getTypeConverter()->convertBlockSignature(dest))
          dest = rewriter.applySignatureConversion(dest, *conversion,
                                                   getTypeConverter());
      std::optional<aiir::ValueRange> cmpOps =
          *caseOp.getCompareOperands(adaptor.getOperands(), t);
      aiir::Attribute attr = cases[t];
      assert(aiir::isa<aiir::UnitAttr>(attr) || cmpOps.has_value());
      if (aiir::isa<fir::PointIntervalAttr>(attr)) {
        auto cmp = aiir::LLVM::ICmpOp::create(rewriter, loc,
                                              aiir::LLVM::ICmpPredicate::eq,
                                              selector, cmpOps->front());
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (aiir::isa<fir::LowerBoundAttr>(attr)) {
        auto cmp = aiir::LLVM::ICmpOp::create(rewriter, loc,
                                              aiir::LLVM::ICmpPredicate::sle,
                                              cmpOps->front(), selector);
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (aiir::isa<fir::UpperBoundAttr>(attr)) {
        auto cmp = aiir::LLVM::ICmpOp::create(rewriter, loc,
                                              aiir::LLVM::ICmpPredicate::sle,
                                              selector, cmpOps->front());
        genCaseLadderStep(loc, cmp, dest, destOps, rewriter);
        continue;
      }
      if (aiir::isa<fir::ClosedIntervalAttr>(attr)) {
        aiir::Value caseArg0 = *cmpOps->begin();
        auto cmp0 = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::sle, caseArg0, selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = createBlock(rewriter, dest);
        auto *newBlock2 = createBlock(rewriter, dest);
        rewriter.setInsertionPointToEnd(thisBlock);
        aiir::LLVM::CondBrOp::create(rewriter, loc, cmp0, newBlock1, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock1);
        aiir::Value caseArg1 = *(cmpOps->begin() + 1);
        auto cmp1 = aiir::LLVM::ICmpOp::create(
            rewriter, loc, aiir::LLVM::ICmpPredicate::sle, selector, caseArg1);
        genCondBrOp(loc, cmp1, dest, destOps, rewriter, newBlock2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(aiir::isa<aiir::UnitAttr>(attr));
      assert((t + 1 == conds) && "unit must be last");
      genBrOp(caseOp, dest, destOps, rewriter);
    }
    return aiir::success();
  }
};

/// Base class for SelectOpConversion and SelectRankOpConversion.
template <typename OP>
struct SelectOpConversionBase : public fir::FIROpConversion<OP> {
  using fir::FIROpConversion<OP>::FIROpConversion;

private:
  /// Helper function for converting select ops. This function converts the
  /// signature of the given block. If the new block signature is different from
  /// `expectedTypes`, returns "failure".
  llvm::FailureOr<aiir::Block *>
  getConvertedBlock(aiir::ConversionPatternRewriter &rewriter,
                    aiir::Operation *branchOp, aiir::Block *block,
                    aiir::TypeRange expectedTypes) const {
    const aiir::TypeConverter *converter = this->getTypeConverter();
    assert(converter && "expected non-null type converter");
    assert(!block->isEntryBlock() && "entry blocks have no predecessors");

    // There is nothing to do if the types already match.
    if (block->getArgumentTypes() == expectedTypes)
      return block;

    // Compute the new block argument types and convert the block.
    std::optional<aiir::TypeConverter::SignatureConversion> conversion =
        converter->convertBlockSignature(block);
    if (!conversion)
      return rewriter.notifyMatchFailure(branchOp,
                                         "could not compute block signature");
    if (expectedTypes != conversion->getConvertedTypes())
      return rewriter.notifyMatchFailure(branchOp,
                                         "mismatch between adaptor operand "
                                         "types and computed block signature");
    return rewriter.applySignatureConversion(block, *conversion, converter);
  }

protected:
  llvm::LogicalResult
  selectMatchAndRewrite(OP select, typename OP::Adaptor adaptor,
                        aiir::ConversionPatternRewriter &rewriter) const {
    unsigned conds = select.getNumConditions();
    auto cases = select.getCases().getValue();
    aiir::Value selector = adaptor.getSelector();
    auto loc = select.getLoc();
    assert(conds > 0 && "select must have cases");

    llvm::SmallVector<aiir::Block *> destinations;
    llvm::SmallVector<aiir::ValueRange> destinationsOperands;
    aiir::Block *defaultDestination;
    aiir::ValueRange defaultOperands;
    // LLVM::SwitchOp selector type and the case values types
    // must have the same bit width, so cast the selector to i64,
    // and use i64 for the case values. It is hard to imagine
    // a computed GO TO with the number of labels in the label-list
    // bigger than INT_MAX, but let's use i64 to be on the safe side.
    // Moreover, fir.select operation is more relaxed than
    // a Fortran computed GO TO, so it may specify such a case value
    // even if there is just a single label/case.
    llvm::SmallVector<int64_t> caseValues;

    for (unsigned t = 0; t != conds; ++t) {
      aiir::Block *dest = select.getSuccessor(t);
      auto destOps = select.getSuccessorOperands(adaptor.getOperands(), t);
      const aiir::Attribute &attr = cases[t];
      if (auto intAttr = aiir::dyn_cast<aiir::IntegerAttr>(attr)) {
        destinationsOperands.push_back(destOps ? *destOps : aiir::ValueRange{});
        auto convertedBlock =
            getConvertedBlock(rewriter, select, dest,
                              aiir::TypeRange(destinationsOperands.back()));
        if (aiir::failed(convertedBlock))
          return aiir::failure();
        destinations.push_back(*convertedBlock);
        caseValues.push_back(intAttr.getInt());
        continue;
      }
      assert(aiir::dyn_cast_or_null<aiir::UnitAttr>(attr));
      assert((t + 1 == conds) && "unit must be last");
      defaultOperands = destOps ? *destOps : aiir::ValueRange{};
      auto convertedBlock = getConvertedBlock(rewriter, select, dest,
                                              aiir::TypeRange(defaultOperands));
      if (aiir::failed(convertedBlock))
        return aiir::failure();
      defaultDestination = *convertedBlock;
    }

    // Deal with the case where there is only a default destination.  Handle it
    // now because emitting empty case values is not legal.
    if (caseValues.empty()) {
      rewriter.replaceOpWithNewOp<aiir::LLVM::BrOp>(select, defaultOperands,
                                                    defaultDestination);
      return aiir::success();
    }

    selector =
        this->integerCast(loc, rewriter, rewriter.getI64Type(), selector);

    rewriter.replaceOpWithNewOp<aiir::LLVM::SwitchOp>(
        select, selector,
        /*defaultDestination=*/defaultDestination,
        /*defaultOperands=*/defaultOperands,
        /*caseValues=*/rewriter.getI64VectorAttr(caseValues),
        /*caseDestinations=*/destinations,
        /*caseOperands=*/destinationsOperands,
        /*branchWeights=*/llvm::ArrayRef<std::int32_t>());
    return aiir::success();
  }
};
/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public SelectOpConversionBase<fir::SelectOp> {
  using SelectOpConversionBase::SelectOpConversionBase;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    return this->selectMatchAndRewrite(op, adaptor, rewriter);
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion
    : public SelectOpConversionBase<fir::SelectRankOp> {
  using SelectOpConversionBase::SelectOpConversionBase;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectRankOp op, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    return this->selectMatchAndRewrite(op, adaptor, rewriter);
  }
};

/// Lower `fir.select_type` to LLVM IR dialect.
struct SelectTypeOpConversion : public fir::FIROpConversion<fir::SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::SelectTypeOp select, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::emitError(select.getLoc(),
                    "fir.select_type should have already been converted");
    return aiir::failure();
  }
};

/// `fir.store` --> `llvm.store`
struct StoreOpConversion : public fir::FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::StoreOp store, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = store.getLoc();
    aiir::Type storeTy = store.getValue().getType();
    aiir::Value llvmValue = adaptor.getValue();
    aiir::Value llvmMemref = adaptor.getMemref();
    aiir::LLVM::AliasAnalysisOpInterface newOp;
    const bool isVolatile =
        fir::isa_volatile_type(store.getMemref().getType()) ||
        fir::isa_volatile_type(store.getValue().getType());
    if (auto boxTy = aiir::dyn_cast<fir::BaseBoxType>(storeTy)) {
      aiir::Type llvmBoxTy = lowerTy().convertBoxTypeAsStruct(boxTy);
      // Always use memcpy because LLVM is not as effective at optimizing
      // aggregate loads/stores as it is optimizing memcpy.
      TypePair boxTypePair{boxTy, llvmBoxTy};
      aiir::Value boxSize =
          computeBoxSize(loc, boxTypePair, llvmValue, rewriter);
      newOp = aiir::LLVM::MemcpyOp::create(rewriter, loc, llvmMemref, llvmValue,
                                           boxSize, isVolatile);
    } else {
      aiir::LLVM::StoreOp storeOp =
          aiir::LLVM::StoreOp::create(rewriter, loc, llvmValue, llvmMemref);

      if (isVolatile)
        storeOp.setVolatile_(true);

      if (store.getNontemporal())
        storeOp.setNontemporal(true);

      if (std::optional<aiir::ArrayAttr> optionalAccessGroups =
              store.getAccessGroups())
        storeOp.setAccessGroups(*optionalAccessGroups);

      newOp = storeOp;
    }
    if (std::optional<aiir::ArrayAttr> optionalTag = store.getTbaa())
      newOp.setTBAATags(*optionalTag);
    else
      attachTBAATag(newOp, storeTy, storeTy, nullptr);
    rewriter.eraseOp(store);
    return aiir::success();
  }
};

/// `fir.copy` --> `llvm.memcpy` or `llvm.memmove`
struct CopyOpConversion : public fir::FIROpConversion<fir::CopyOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::CopyOp copy, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Location loc = copy.getLoc();
    const bool isVolatile =
        fir::isa_volatile_type(copy.getSource().getType()) ||
        fir::isa_volatile_type(copy.getDestination().getType());
    aiir::Value llvmSource = adaptor.getSource();
    aiir::Value llvmDestination = adaptor.getDestination();
    aiir::Type i64Ty = aiir::IntegerType::get(rewriter.getContext(), 64);
    aiir::Type copyTy = fir::unwrapRefType(copy.getSource().getType());
    aiir::Value copySize = genTypeStrideInBytes(
        loc, i64Ty, rewriter, convertType(copyTy), getDataLayout());

    aiir::LLVM::AliasAnalysisOpInterface newOp;
    if (copy.getNoOverlap())
      newOp = aiir::LLVM::MemcpyOp::create(rewriter, loc, llvmDestination,
                                           llvmSource, copySize, isVolatile);
    else
      newOp = aiir::LLVM::MemmoveOp::create(rewriter, loc, llvmDestination,
                                            llvmSource, copySize, isVolatile);

    // TODO: propagate TBAA once FirAliasTagOpInterface added to CopyOp.
    attachTBAATag(newOp, copyTy, copyTy, nullptr);
    rewriter.eraseOp(copy);
    return aiir::success();
  }
};

namespace {

/// Convert `fir.unboxchar` into two `llvm.extractvalue` instructions. One for
/// the character buffer and one for the buffer length.
struct UnboxCharOpConversion : public fir::FIROpConversion<fir::UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UnboxCharOp unboxchar, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type lenTy = convertType(unboxchar.getType(1));
    aiir::Value tuple = adaptor.getOperands()[0];

    aiir::Location loc = unboxchar.getLoc();
    aiir::Value ptrToBuffer =
        aiir::LLVM::ExtractValueOp::create(rewriter, loc, tuple, 0);

    auto len = aiir::LLVM::ExtractValueOp::create(rewriter, loc, tuple, 1);
    aiir::Value lenAfterCast = integerCast(loc, rewriter, lenTy, len);

    rewriter.replaceOp(unboxchar,
                       llvm::ArrayRef<aiir::Value>{ptrToBuffer, lenAfterCast});
    return aiir::success();
  }
};

/// Lower `fir.unboxproc` operation. Unbox a procedure box value, yielding its
/// components.
/// TODO: Part of supporting Fortran 2003 procedure pointers.
struct UnboxProcOpConversion : public fir::FIROpConversion<fir::UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UnboxProcOp unboxproc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    TODO(unboxproc.getLoc(), "fir.unboxproc codegen");
    return aiir::failure();
  }
};

/// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public fir::FIROpConversion<fir::UndefOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UndefOp undef, OpAdaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    if (aiir::isa<fir::DummyScopeType>(undef.getType())) {
      // Dummy scoping is used for Fortran analyses like AA. Once it gets to
      // pre-codegen rewrite it is erased and a fir.undef is created to
      // feed to the fir declare operation. Thus, during codegen, we can
      // simply erase is as it is no longer used.
      rewriter.eraseOp(undef);
      return aiir::success();
    }
    rewriter.replaceOpWithNewOp<aiir::LLVM::UndefOp>(
        undef, convertType(undef.getType()));
    return aiir::success();
  }
};

struct ZeroOpConversion : public fir::FIROpConversion<fir::ZeroOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::ZeroOp zero, OpAdaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type ty = convertType(zero.getType());
    rewriter.replaceOpWithNewOp<aiir::LLVM::ZeroOp>(zero, ty);
    return aiir::success();
  }
};

/// `fir.unreachable` --> `llvm.unreachable`
struct UnreachableOpConversion
    : public fir::FIROpConversion<fir::UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::UnreachableOp unreach, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<aiir::LLVM::UnreachableOp>(unreach);
    return aiir::success();
  }
};

/// `fir.is_present` -->
/// ```
///  %0 = llvm.aiir.constant(0 : i64)
///  %1 = llvm.ptrtoint %0
///  %2 = llvm.icmp "ne" %1, %0 : i64
/// ```
struct IsPresentOpConversion : public fir::FIROpConversion<fir::IsPresentOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::IsPresentOp isPresent, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type idxTy = lowerTy().indexType();
    aiir::Location loc = isPresent.getLoc();
    auto ptr = adaptor.getOperands()[0];

    if (aiir::isa<fir::BoxCharType>(isPresent.getVal().getType())) {
      [[maybe_unused]] auto structTy =
          aiir::cast<aiir::LLVM::LLVMStructType>(ptr.getType());
      assert(!structTy.isOpaque() && !structTy.getBody().empty());

      ptr = aiir::LLVM::ExtractValueOp::create(rewriter, loc, ptr, 0);
    }
    aiir::LLVM::ConstantOp c0 =
        fir::genConstantIndex(isPresent.getLoc(), idxTy, rewriter, 0);
    auto addr = aiir::LLVM::PtrToIntOp::create(rewriter, loc, idxTy, ptr);
    rewriter.replaceOpWithNewOp<aiir::LLVM::ICmpOp>(
        isPresent, aiir::LLVM::ICmpPredicate::ne, addr, c0);

    return aiir::success();
  }
};

/// Create value signaling an absent optional argument in a call, e.g.
/// `fir.absent !fir.ref<i64>` -->  `llvm.aiir.zero : !llvm.ptr<i64>`
struct AbsentOpConversion : public fir::FIROpConversion<fir::AbsentOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AbsentOp absent, OpAdaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    aiir::Type ty = convertType(absent.getType());
    rewriter.replaceOpWithNewOp<aiir::LLVM::ZeroOp>(absent, ty);
    return aiir::success();
  }
};

//
// Primitive operations on Complex types
//

template <typename OPTY>
static inline aiir::LLVM::FastmathFlagsAttr getLLVMFMFAttr(OPTY op) {
  return aiir::LLVM::FastmathFlagsAttr::get(
      op.getContext(),
      aiir::arith::convertArithFastMathFlagsToLLVM(op.getFastmath()));
}

/// Generate inline code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
static aiir::LLVM::InsertValueOp
complexSum(OPTY sumop, aiir::ValueRange opnds,
           aiir::ConversionPatternRewriter &rewriter,
           const fir::LLVMTypeConverter &lowering) {
  aiir::LLVM::FastmathFlagsAttr fmf = getLLVMFMFAttr(sumop);
  aiir::Value a = opnds[0];
  aiir::Value b = opnds[1];
  auto loc = sumop.getLoc();
  aiir::Type eleTy = lowering.convertType(getComplexEleTy(sumop.getType()));
  aiir::Type ty = lowering.convertType(sumop.getType());
  auto x0 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, a, 0);
  auto y0 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, a, 1);
  auto x1 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, b, 0);
  auto y1 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, b, 1);
  auto rx = LLVMOP::create(rewriter, loc, eleTy, x0, x1, fmf);
  auto ry = LLVMOP::create(rewriter, loc, eleTy, y0, y1, fmf);
  auto r0 = aiir::LLVM::UndefOp::create(rewriter, loc, ty);
  llvm::SmallVector<int64_t> pos{0};
  auto r1 = aiir::LLVM::InsertValueOp::create(rewriter, loc, r0, rx, pos);
  return aiir::LLVM::InsertValueOp::create(rewriter, loc, r1, ry, 1);
}
} // namespace

namespace {
struct AddcOpConversion : public fir::FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::AddcOp addc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) + (x' + iy')
    // result: (x + x') + i(y + y')
    auto r = complexSum<aiir::LLVM::FAddOp>(addc, adaptor.getOperands(),
                                            rewriter, lowerTy());
    rewriter.replaceOp(addc, r.getResult());
    return aiir::success();
  }
};

struct SubcOpConversion : public fir::FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::SubcOp subc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // given: (x + iy) - (x' + iy')
    // result: (x - x') + i(y - y')
    auto r = complexSum<aiir::LLVM::FSubOp>(subc, adaptor.getOperands(),
                                            rewriter, lowerTy());
    rewriter.replaceOp(subc, r.getResult());
    return aiir::success();
  }
};

/// Inlined complex multiply
struct MulcOpConversion : public fir::FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::MulcOp mulc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // TODO: Can we use a call to __muldc3 ?
    // given: (x + iy) * (x' + iy')
    // result: (xx'-yy')+i(xy'+yx')
    aiir::LLVM::FastmathFlagsAttr fmf = getLLVMFMFAttr(mulc);
    aiir::Value a = adaptor.getOperands()[0];
    aiir::Value b = adaptor.getOperands()[1];
    auto loc = mulc.getLoc();
    aiir::Type eleTy = convertType(getComplexEleTy(mulc.getType()));
    aiir::Type ty = convertType(mulc.getType());
    auto x0 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, a, 0);
    auto y0 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, a, 1);
    auto x1 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, b, 0);
    auto y1 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, b, 1);
    auto xx = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, x0, x1, fmf);
    auto yx = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, y0, x1, fmf);
    auto xy = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, x0, y1, fmf);
    auto ri = aiir::LLVM::FAddOp::create(rewriter, loc, eleTy, xy, yx, fmf);
    auto yy = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, y0, y1, fmf);
    auto rr = aiir::LLVM::FSubOp::create(rewriter, loc, eleTy, xx, yy, fmf);
    auto ra = aiir::LLVM::UndefOp::create(rewriter, loc, ty);
    llvm::SmallVector<int64_t> pos{0};
    auto r1 = aiir::LLVM::InsertValueOp::create(rewriter, loc, ra, rr, pos);
    auto r0 = aiir::LLVM::InsertValueOp::create(rewriter, loc, r1, ri, 1);
    rewriter.replaceOp(mulc, r0.getResult());
    return aiir::success();
  }
};

/// Inlined complex division
struct DivcOpConversion : public fir::FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::DivcOp divc, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // TODO: Can we use a call to __divdc3 instead?
    // Just generate inline code for now.
    // given: (x + iy) / (x' + iy')
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    aiir::LLVM::FastmathFlagsAttr fmf = getLLVMFMFAttr(divc);
    aiir::Value a = adaptor.getOperands()[0];
    aiir::Value b = adaptor.getOperands()[1];
    auto loc = divc.getLoc();
    aiir::Type eleTy = convertType(getComplexEleTy(divc.getType()));
    aiir::Type ty = convertType(divc.getType());
    auto x0 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, a, 0);
    auto y0 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, a, 1);
    auto x1 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, b, 0);
    auto y1 = aiir::LLVM::ExtractValueOp::create(rewriter, loc, b, 1);
    auto xx = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, x0, x1, fmf);
    auto x1x1 = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, x1, x1, fmf);
    auto yx = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, y0, x1, fmf);
    auto xy = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, x0, y1, fmf);
    auto yy = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, y0, y1, fmf);
    auto y1y1 = aiir::LLVM::FMulOp::create(rewriter, loc, eleTy, y1, y1, fmf);
    auto d = aiir::LLVM::FAddOp::create(rewriter, loc, eleTy, x1x1, y1y1, fmf);
    auto rrn = aiir::LLVM::FAddOp::create(rewriter, loc, eleTy, xx, yy, fmf);
    auto rin = aiir::LLVM::FSubOp::create(rewriter, loc, eleTy, yx, xy, fmf);
    auto rr = aiir::LLVM::FDivOp::create(rewriter, loc, eleTy, rrn, d, fmf);
    auto ri = aiir::LLVM::FDivOp::create(rewriter, loc, eleTy, rin, d, fmf);
    auto ra = aiir::LLVM::UndefOp::create(rewriter, loc, ty);
    llvm::SmallVector<int64_t> pos{0};
    auto r1 = aiir::LLVM::InsertValueOp::create(rewriter, loc, ra, rr, pos);
    auto r0 = aiir::LLVM::InsertValueOp::create(rewriter, loc, r1, ri, 1);
    rewriter.replaceOp(divc, r0.getResult());
    return aiir::success();
  }
};

/// Inlined complex negation
struct NegcOpConversion : public fir::FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::NegcOp neg, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {
    // given: -(x + iy)
    // result: -x - iy
    auto eleTy = convertType(getComplexEleTy(neg.getType()));
    auto loc = neg.getLoc();
    aiir::Value o0 = adaptor.getOperands()[0];
    auto rp = aiir::LLVM::ExtractValueOp::create(rewriter, loc, o0, 0);
    auto ip = aiir::LLVM::ExtractValueOp::create(rewriter, loc, o0, 1);
    auto nrp = aiir::LLVM::FNegOp::create(rewriter, loc, eleTy, rp);
    auto nip = aiir::LLVM::FNegOp::create(rewriter, loc, eleTy, ip);
    llvm::SmallVector<int64_t> pos{0};
    auto r = aiir::LLVM::InsertValueOp::create(rewriter, loc, o0, nrp, pos);
    rewriter.replaceOpWithNewOp<aiir::LLVM::InsertValueOp>(neg, r, nip, 1);
    return aiir::success();
  }
};

struct BoxOffsetOpConversion : public fir::FIROpConversion<fir::BoxOffsetOp> {
  using FIROpConversion::FIROpConversion;

  llvm::LogicalResult
  matchAndRewrite(fir::BoxOffsetOp boxOffset, OpAdaptor adaptor,
                  aiir::ConversionPatternRewriter &rewriter) const override {

    aiir::Type pty = ::getLlvmPtrType(boxOffset.getContext());
    aiir::Type boxRefType = fir::unwrapRefType(boxOffset.getBoxRef().getType());

    assert((aiir::isa<fir::BaseBoxType>(boxRefType) ||
            aiir::isa<fir::BoxCharType>(boxRefType)) &&
           "boxRef should be a reference to either fir.box or fir.boxchar");

    aiir::Type llvmBoxTy;
    int fieldId;
    if (auto boxType = aiir::dyn_cast_or_null<fir::BaseBoxType>(boxRefType)) {
      llvmBoxTy = lowerTy().convertBoxTypeAsStruct(
          aiir::cast<fir::BaseBoxType>(boxType));
      fieldId = boxOffset.getField() == fir::BoxFieldAttr::derived_type
                    ? getTypeDescFieldId(boxType)
                    : kAddrPosInBox;
    } else {
      auto boxCharType = aiir::cast<fir::BoxCharType>(boxRefType);
      llvmBoxTy = lowerTy().convertType(boxCharType);
      fieldId = kAddrPosInBox;
    }
    rewriter.replaceOpWithNewOp<aiir::LLVM::GEPOp>(
        boxOffset, pty, llvmBoxTy, adaptor.getBoxRef(),
        llvm::ArrayRef<aiir::LLVM::GEPArg>{0, fieldId});
    return aiir::success();
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
                  aiir::ConversionPatternRewriter &rewriter) const final {
    if (!op->getUses().empty())
      return rewriter.notifyMatchFailure(op, "op must be dead");
    rewriter.eraseOp(op);
    return aiir::success();
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
    : public aiir::OpRewritePattern<aiir::LLVM::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(aiir::LLVM::CallOp op,
                  aiir::PatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    auto callee = op.getCallee();
    if (callee)
      if (*callee == "hypotf")
        op.setCalleeAttr(aiir::SymbolRefAttr::get(op.getContext(), "_hypotf"));

    rewriter.finalizeOpModification(op);
    return aiir::success();
  }
};

class RenameMSVCLibmFuncs
    : public aiir::OpRewritePattern<aiir::LLVM::LLVMFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  llvm::LogicalResult
  matchAndRewrite(aiir::LLVM::LLVMFuncOp op,
                  aiir::PatternRewriter &rewriter) const override {
    rewriter.startOpModification(op);
    if (op.getSymName() == "hypotf")
      op.setSymNameAttr(rewriter.getStringAttr("_hypotf"));
    rewriter.finalizeOpModification(op);
    return aiir::success();
  }
};
} // namespace

namespace {
/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect. An
/// AIIR pass is used to lower residual Std dialect to LLVM IR dialect.
class FIRToLLVMLowering
    : public fir::impl::FIRToLLVMLoweringBase<FIRToLLVMLowering> {
public:
  FIRToLLVMLowering() = default;
  FIRToLLVMLowering(fir::FIRToLLVMPassOptions options) : options{options} {}
  aiir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto mod = getModule();
    if (!forcedTargetTriple.empty())
      fir::setTargetTriple(mod, forcedTargetTriple);

    if (!forcedDataLayout.empty()) {
      llvm::DataLayout dl(forcedDataLayout);
      fir::support::setAIIRDataLayout(mod, dl);
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
    aiir::OpPassManager mathConversionPM("builtin.module");

    bool isAMDGCN = fir::getTargetTriple(mod).isAMDGCN();
    bool isNVPTX = fir::getTargetTriple(mod).isNVPTX();
    // If compiling for AMD target some math operations must be lowered to AMD
    // GPU library calls, the rest can be converted to LLVM intrinsics, which
    // is handled in the mathToLLVM conversion. The lowering to libm calls is
    // not needed since all math operations are handled this way.
    if (isAMDGCN) {
      mathConversionPM.addPass(aiir::createConvertMathToROCDL());
      mathConversionPM.addPass(aiir::createConvertComplexToROCDLLibraryCalls());
    }
    // If compiling for NVIDIA target some math operations must be lowered to
    // NVVM libdevice calls.
    if (isNVPTX)
      mathConversionPM.addPass(aiir::createConvertMathToNVVM());

    // Convert math::FPowI operations to inline implementation
    // only if the exponent's width is greater than 32, otherwise,
    // it will be lowered to LLVM intrinsic operation by a later conversion.
    aiir::ConvertMathToFuncsOptions mathToFuncsOptions{};
    mathToFuncsOptions.minWidthOfFPowIExponent = 33;
    mathConversionPM.addPass(
        aiir::createConvertMathToFuncs(mathToFuncsOptions));

    aiir::ConvertComplexToStandardPassOptions complexToStandardOptions{};
    if (options.ComplexRange ==
        Fortran::frontend::CodeGenOptions::ComplexRangeKind::CX_Basic) {
      complexToStandardOptions.complexRange =
          aiir::complex::ComplexRangeFlags::basic;
    } else if (options.ComplexRange == Fortran::frontend::CodeGenOptions::
                                           ComplexRangeKind::CX_Improved) {
      complexToStandardOptions.complexRange =
          aiir::complex::ComplexRangeFlags::improved;
    }
    mathConversionPM.addPass(
        aiir::createConvertComplexToStandardPass(complexToStandardOptions));

    // Convert Math dialect operations into LLVM dialect operations.
    // There is no way to prefer MathToLLVM patterns over MathToLibm
    // patterns (applied below), so we have to run MathToLLVM conversion here.
    mathConversionPM.addNestedPass<aiir::func::FuncOp>(
        aiir::createConvertMathToLLVMPass());
    if (aiir::failed(runPipeline(mathConversionPM, mod)))
      return signalPassFailure();

    std::optional<aiir::DataLayout> dl =
        fir::support::getOrSetAIIRDataLayout(mod, /*allowDefaultLayout=*/true);
    if (!dl) {
      aiir::emitError(mod.getLoc(),
                      "module operation must carry a data layout attribute "
                      "to generate llvm IR from FIR");
      signalPassFailure();
      return;
    }

    auto *context = getModule().getContext();
    fir::LLVMTypeConverter typeConverter{getModule(),
                                         options.applyTBAA || applyTBAA,
                                         options.forceUnifiedTBAATree, *dl};
    aiir::RewritePatternSet pattern(context);
    fir::populateFIRToLLVMConversionPatterns(typeConverter, pattern, options);
    aiir::populateFuncToLLVMConversionPatterns(typeConverter, pattern);
    aiir::populateOpenMPToLLVMConversionPatterns(typeConverter, pattern);
    aiir::arith::populateArithToLLVMConversionPatterns(typeConverter, pattern);
    aiir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          pattern);
    aiir::cf::populateAssertToLLVMConversionPattern(typeConverter, pattern);
    // Math operations that have not been converted yet must be converted
    // to Libm.
    if (!isAMDGCN && !isNVPTX)
      aiir::populateMathToLibmConversionPatterns(pattern);
    aiir::populateComplexToLLVMConversionPatterns(typeConverter, pattern);
    aiir::index::populateIndexToLLVMConversionPatterns(typeConverter, pattern);
    aiir::populateVectorToLLVMConversionPatterns(typeConverter, pattern);

    // Flang specific overloads for OpenMP operations, to allow for special
    // handling of things like Box types.
    fir::populateOpenMPFIRToLLVMConversionPatterns(typeConverter, pattern);

    aiir::ConversionTarget target{*context};
    target.addLegalDialect<aiir::LLVM::LLVMDialect>();
    // The OpenMP dialect is legal for Operations without regions, for those
    // which contains regions it is legal if the region contains only the
    // LLVM dialect. Add OpenMP dialect as a legal dialect for conversion and
    // legalize conversion of OpenMP operations without regions.
    aiir::configureOpenMPToLLVMConversionLegality(target, typeConverter);
    target.addLegalDialect<aiir::omp::OpenMPDialect>();
    target.addLegalDialect<aiir::acc::OpenACCDialect>();
    target.addLegalDialect<aiir::gpu::GPUDialect>();

    // required NOPs for applying a full conversion
    target.addLegalOp<aiir::ModuleOp>();

    // If we're on Windows, we might need to rename some libm calls.
    bool isMSVC = fir::getTargetTriple(mod).isOSMSVCRT();
    if (isMSVC) {
      pattern.insert<RenameMSVCLibmCallees, RenameMSVCLibmFuncs>(context);

      target.addDynamicallyLegalOp<aiir::LLVM::CallOp>(
          [](aiir::LLVM::CallOp op) {
            auto callee = op.getCallee();
            if (!callee)
              return true;
            return *callee != "hypotf";
          });
      target.addDynamicallyLegalOp<aiir::LLVM::LLVMFuncOp>(
          [](aiir::LLVM::LLVMFuncOp op) {
            return op.getSymName() != "hypotf";
          });
    }

    // apply the patterns
    if (aiir::failed(aiir::applyFullConversion(getModule(), target,
                                               std::move(pattern)))) {
      signalPassFailure();
    }

    // Run pass to add comdats to functions that have weak linkage on relevant
    // platforms
    if (fir::getTargetTriple(mod).supportsCOMDAT()) {
      aiir::OpPassManager comdatPM("builtin.module");
      comdatPM.addPass(aiir::LLVM::createLLVMAddComdats());
      if (aiir::failed(runPipeline(comdatPM, mod)))
        return signalPassFailure();
    }
  }

private:
  fir::FIRToLLVMPassOptions options;
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass
    : public aiir::PassWrapper<LLVMIRLoweringPass,
                               aiir::OperationPass<aiir::ModuleOp>> {
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMIRLoweringPass)

  LLVMIRLoweringPass(llvm::raw_ostream &output, fir::LLVMIRLoweringPrinter p)
      : output{output}, printer{p} {}

  aiir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto *ctx = getModule().getContext();
    auto optName = getModule().getName();
    llvm::LLVMContext llvmCtx;
    if (auto llvmModule = aiir::translateModuleToLLVMIR(
            getModule(), llvmCtx, optName ? *optName : "FIRModule")) {
      printer(*llvmModule, output);
      return;
    }

    aiir::emitError(aiir::UnknownLoc::get(ctx), "could not emit LLVM-IR\n");
    signalPassFailure();
  }

private:
  llvm::raw_ostream &output;
  fir::LLVMIRLoweringPrinter printer;
};

} // namespace

std::unique_ptr<aiir::Pass> fir::createFIRToLLVMPass() {
  return std::make_unique<FIRToLLVMLowering>();
}

std::unique_ptr<aiir::Pass>
fir::createFIRToLLVMPass(fir::FIRToLLVMPassOptions options) {
  return std::make_unique<FIRToLLVMLowering>(options);
}

std::unique_ptr<aiir::Pass>
fir::createLLVMDialectToLLVMPass(llvm::raw_ostream &output,
                                 fir::LLVMIRLoweringPrinter printer) {
  return std::make_unique<LLVMIRLoweringPass>(output, printer);
}

void fir::populateFIRToLLVMConversionPatterns(
    const fir::LLVMTypeConverter &converter, aiir::RewritePatternSet &patterns,
    fir::FIRToLLVMPassOptions &options) {
  patterns.insert<
      AbsentOpConversion, AddcOpConversion, AddrOfOpConversion,
      AllocaOpConversion, AllocMemOpConversion, BitcastOpConversion,
      BoxAddrOpConversion, BoxCharLenOpConversion, BoxDimsOpConversion,
      BoxEleSizeOpConversion, BoxIsAllocOpConversion, BoxIsArrayOpConversion,
      BoxIsPtrOpConversion, AssumedSizeExtentOpConversion,
      IsAssumedSizeExtentOpConversion, BoxOffsetOpConversion,
      BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeCodeOpConversion,
      BoxTypeDescOpConversion, CallOpConversion, CmpcOpConversion,
      VolatileCastOpConversion, ConvertOpConversion, CoordinateOpConversion,
      CopyOpConversion, DTEntryOpConversion, DeclareOpConversion,
      DeclareValueOpConversion,
      DoConcurrentSpecifierOpConversion<fir::LocalitySpecifierOp>,
      DoConcurrentSpecifierOpConversion<fir::DeclareReductionOp>,
      DivcOpConversion, EmboxOpConversion, EmboxCharOpConversion,
      EmboxProcOpConversion, ExtractValueOpConversion, FieldIndexOpConversion,
      FirEndOpConversion, FreeMemOpConversion, GlobalLenOpConversion,
      GlobalOpConversion, InsertOnRangeOpConversion, IsPresentOpConversion,
      LenParamIndexOpConversion, LoadOpConversion, MulcOpConversion,
      NegcOpConversion, NoReassocOpConversion, PrefetchOpConversion,
      SelectCaseOpConversion, SelectOpConversion, SelectRankOpConversion,
      SelectTypeOpConversion, ShapeOpConversion, ShapeShiftOpConversion,
      ShiftOpConversion, SliceOpConversion, StoreOpConversion,
      StringLitOpConversion, SubcOpConversion, TypeDescOpConversion,
      TypeInfoOpConversion, UnboxCharOpConversion, UnboxProcOpConversion,
      UndefOpConversion, UnreachableOpConversion, UseStmtOpConversion,
      XArrayCoorOpConversion, XEmboxOpConversion, XReboxOpConversion,
      ZeroOpConversion>(converter, options);

  // Patterns that are populated without a type converter do not trigger
  // target materializations for the operands of the root op.
  patterns.insert<HasValueOpConversion, InsertValueOpConversion>(
      patterns.getContext());
}
