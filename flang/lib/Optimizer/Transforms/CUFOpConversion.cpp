//===-- CUFDeviceGlobal.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Transforms/CUFOpConversion.h"
#include "flang/Common/Fortran.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/allocatable.h"
#include "flang/Runtime/CUDA/common.h"
#include "flang/Runtime/CUDA/descriptor.h"
#include "flang/Runtime/CUDA/memory.h"
#include "flang/Runtime/allocatable.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace fir {
#define GEN_PASS_DEF_CUFOPCONVERSION
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace fir;
using namespace mlir;
using namespace Fortran::runtime;
using namespace Fortran::runtime::cuda;

namespace {

static inline unsigned getMemType(cuf::DataAttribute attr) {
  if (attr == cuf::DataAttribute::Device)
    return kMemTypeDevice;
  if (attr == cuf::DataAttribute::Managed)
    return kMemTypeManaged;
  if (attr == cuf::DataAttribute::Unified)
    return kMemTypeUnified;
  if (attr == cuf::DataAttribute::Pinned)
    return kMemTypePinned;
  llvm::report_fatal_error("unsupported memory type");
}

template <typename OpTy>
static bool isPinned(OpTy op) {
  if (op.getDataAttr() && *op.getDataAttr() == cuf::DataAttribute::Pinned)
    return true;
  return false;
}

template <typename OpTy>
static bool hasDoubleDescriptors(OpTy op) {
  if (auto declareOp =
          mlir::dyn_cast_or_null<fir::DeclareOp>(op.getBox().getDefiningOp())) {
    if (mlir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp())) {
      if (isPinned(declareOp))
        return false;
      return true;
    }
  } else if (auto declareOp = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
                 op.getBox().getDefiningOp())) {
    if (mlir::isa_and_nonnull<fir::AddrOfOp>(
            declareOp.getMemref().getDefiningOp())) {
      if (isPinned(declareOp))
        return false;
      return true;
    }
  }
  return false;
}

static mlir::Value createConvertOp(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val) {
  if (val.getType() != toTy)
    return rewriter.create<fir::ConvertOp>(loc, toTy, val);
  return val;
}

template <typename OpTy>
static mlir::LogicalResult convertOpToCall(OpTy op,
                                           mlir::PatternRewriter &rewriter,
                                           mlir::func::FuncOp func) {
  auto mod = op->template getParentOfType<mlir::ModuleOp>();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Location loc = op.getLoc();
  auto fTy = func.getFunctionType();

  mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
  mlir::Value sourceLine;
  if constexpr (std::is_same_v<OpTy, cuf::AllocateOp>)
    sourceLine = fir::factory::locationToLineNo(
        builder, loc, op.getSource() ? fTy.getInput(6) : fTy.getInput(5));
  else
    sourceLine = fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));

  mlir::Value hasStat = op.getHasStat() ? builder.createBool(loc, true)
                                        : builder.createBool(loc, false);

  mlir::Value errmsg;
  if (op.getErrmsg()) {
    errmsg = op.getErrmsg();
  } else {
    mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
    errmsg = builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  }
  llvm::SmallVector<mlir::Value> args;
  if constexpr (std::is_same_v<OpTy, cuf::AllocateOp>) {
    if (op.getSource()) {
      mlir::Value stream =
          op.getStream()
              ? op.getStream()
              : builder.createIntegerConstant(loc, fTy.getInput(2), -1);
      args = fir::runtime::createArguments(builder, loc, fTy, op.getBox(),
                                           op.getSource(), stream, hasStat,
                                           errmsg, sourceFile, sourceLine);
    } else {
      mlir::Value stream =
          op.getStream()
              ? op.getStream()
              : builder.createIntegerConstant(loc, fTy.getInput(1), -1);
      args = fir::runtime::createArguments(builder, loc, fTy, op.getBox(),
                                           stream, hasStat, errmsg, sourceFile,
                                           sourceLine);
    }
  } else {
    args =
        fir::runtime::createArguments(builder, loc, fTy, op.getBox(), hasStat,
                                      errmsg, sourceFile, sourceLine);
  }
  auto callOp = builder.create<fir::CallOp>(loc, func, args);
  rewriter.replaceOp(op, callOp);
  return mlir::success();
}

struct CUFAllocateOpConversion
    : public mlir::OpRewritePattern<cuf::AllocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // TODO: Pinned is a reference to a logical value that can be set to true
    // when pinned allocation succeed. This will require a new entry point.
    if (op.getPinned())
      return mlir::failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (hasDoubleDescriptors(op)) {
      // Allocation for module variable are done with custom runtime entry point
      // so the descriptors can be synchronized.
      mlir::func::FuncOp func;
      if (op.getSource())
        func = fir::runtime::getRuntimeFunc<mkRTKey(
            CUFAllocatableAllocateSourceSync)>(loc, builder);
      else
        func =
            fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableAllocateSync)>(
                loc, builder);
      return convertOpToCall<cuf::AllocateOp>(op, rewriter, func);
    }

    mlir::func::FuncOp func;
    if (op.getSource())
      func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableAllocateSource)>(
              loc, builder);
    else
      func = fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableAllocate)>(
          loc, builder);

    return convertOpToCall<cuf::AllocateOp>(op, rewriter, func);
  }
};

struct CUFDeallocateOpConversion
    : public mlir::OpRewritePattern<cuf::DeallocateOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::DeallocateOp op,
                  mlir::PatternRewriter &rewriter) const override {

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    if (hasDoubleDescriptors(op)) {
      // Deallocation for module variable are done with custom runtime entry
      // point so the descriptors can be synchronized.
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocatableDeallocate)>(
              loc, builder);
      return convertOpToCall<cuf::DeallocateOp>(op, rewriter, func);
    }

    // Deallocation for local descriptor falls back on the standard runtime
    // AllocatableDeallocate as the dedicated deallocator is set in the
    // descriptor before the call.
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(AllocatableDeallocate)>(loc,
                                                                     builder);
    return convertOpToCall<cuf::DeallocateOp>(op, rewriter, func);
  }
};

static bool inDeviceContext(mlir::Operation *op) {
  if (op->getParentOfType<cuf::KernelOp>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::gpu::GPUFuncOp>())
    return true;
  if (auto funcOp = op->getParentOfType<mlir::func::FuncOp>()) {
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName())) {
      return cudaProcAttr.getValue() != cuf::ProcAttribute::Host &&
             cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice;
    }
  }
  return false;
}

static int computeWidth(mlir::Location loc, mlir::Type type,
                        fir::KindMapping &kindMap) {
  auto eleTy = fir::unwrapSequenceType(type);
  if (auto t{mlir::dyn_cast<mlir::IntegerType>(eleTy)})
    return t.getWidth() / 8;
  if (auto t{mlir::dyn_cast<mlir::FloatType>(eleTy)})
    return t.getWidth() / 8;
  if (eleTy.isInteger(1))
    return 1;
  if (auto t{mlir::dyn_cast<fir::LogicalType>(eleTy)})
    return kindMap.getLogicalBitsize(t.getFKind()) / 8;
  if (auto t{mlir::dyn_cast<mlir::ComplexType>(eleTy)}) {
    int elemSize =
        mlir::cast<mlir::FloatType>(t.getElementType()).getWidth() / 8;
    return 2 * elemSize;
  }
  if (auto t{mlir::dyn_cast_or_null<fir::CharacterType>(eleTy)})
    return kindMap.getCharacterBitsize(t.getFKind()) / 8;
  mlir::emitError(loc, "unsupported type");
  return 0;
}

struct CUFAllocOpConversion : public mlir::OpRewritePattern<cuf::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFAllocOpConversion(mlir::MLIRContext *context, mlir::DataLayout *dl,
                       const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), dl{dl}, typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::AllocOp op,
                  mlir::PatternRewriter &rewriter) const override {

    if (inDeviceContext(op.getOperation())) {
      // In device context just replace the cuf.alloc operation with a fir.alloc
      // the cuf.free will be removed.
      rewriter.replaceOpWithNewOp<fir::AllocaOp>(
          op, op.getInType(), op.getUniqName() ? *op.getUniqName() : "",
          op.getBindcName() ? *op.getBindcName() : "", op.getTypeparams(),
          op.getShape());
      return mlir::success();
    }

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    if (!mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType())) {
      // Convert scalar and known size array allocations.
      mlir::Value bytes;
      fir::KindMapping kindMap{fir::getKindMapping(mod)};
      if (fir::isa_trivial(op.getInType())) {
        int width = computeWidth(loc, op.getInType(), kindMap);
        bytes =
            builder.createIntegerConstant(loc, builder.getIndexType(), width);
      } else if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(
                     op.getInType())) {
        std::size_t size = 0;
        if (fir::isa_derived(seqTy.getEleTy())) {
          mlir::Type structTy = typeConverter->convertType(seqTy.getEleTy());
          size = dl->getTypeSizeInBits(structTy) / 8;
        } else {
          size = computeWidth(loc, seqTy.getEleTy(), kindMap);
        }
        mlir::Value width =
            builder.createIntegerConstant(loc, builder.getIndexType(), size);
        mlir::Value nbElem;
        if (fir::sequenceWithNonConstantShape(seqTy)) {
          assert(!op.getShape().empty() && "expect shape with dynamic arrays");
          nbElem = builder.loadIfRef(loc, op.getShape()[0]);
          for (unsigned i = 1; i < op.getShape().size(); ++i) {
            nbElem = rewriter.create<mlir::arith::MulIOp>(
                loc, nbElem, builder.loadIfRef(loc, op.getShape()[i]));
          }
        } else {
          nbElem = builder.createIntegerConstant(loc, builder.getIndexType(),
                                                 seqTy.getConstantArraySize());
        }
        bytes = rewriter.create<mlir::arith::MulIOp>(loc, nbElem, width);
      } else if (fir::isa_derived(op.getInType())) {
        mlir::Type structTy = typeConverter->convertType(op.getInType());
        std::size_t structSize = dl->getTypeSizeInBits(structTy) / 8;
        bytes = builder.createIntegerConstant(loc, builder.getIndexType(),
                                              structSize);
      } else {
        mlir::emitError(loc, "unsupported type in cuf.alloc\n");
      }
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemAlloc)>(loc, builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      mlir::Value memTy = builder.createIntegerConstant(
          loc, builder.getI32Type(), getMemType(op.getDataAttr()));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, bytes, memTy, sourceFile, sourceLine)};
      auto callOp = builder.create<fir::CallOp>(loc, func, args);
      auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                          callOp.getResult(0));
      rewriter.replaceOp(op, convOp);
      return mlir::success();
    }

    // Convert descriptor allocations to function call.
    auto boxTy = mlir::dyn_cast_or_null<fir::BaseBoxType>(op.getInType());
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFAllocDescriptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));

    mlir::Type structTy = typeConverter->convertBoxTypeAsStruct(boxTy);
    std::size_t boxSize = dl->getTypeSizeInBits(structTy) / 8;
    mlir::Value sizeInBytes =
        builder.createIntegerConstant(loc, builder.getIndexType(), boxSize);

    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, sizeInBytes, sourceFile, sourceLine)};
    auto callOp = builder.create<fir::CallOp>(loc, func, args);
    auto convOp = builder.createConvert(loc, op.getResult().getType(),
                                        callOp.getResult(0));
    rewriter.replaceOp(op, convOp);
    return mlir::success();
  }

private:
  mlir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

struct DeclareOpConversion : public mlir::OpRewritePattern<fir::DeclareOp> {
  using OpRewritePattern::OpRewritePattern;

  DeclareOpConversion(mlir::MLIRContext *context,
                      const mlir::SymbolTable &symtab)
      : OpRewritePattern(context), symTab{symtab} {}

  mlir::LogicalResult
  matchAndRewrite(fir::DeclareOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (auto addrOfOp = op.getMemref().getDefiningOp<fir::AddrOfOp>()) {
      if (auto global = symTab.lookup<fir::GlobalOp>(
              addrOfOp.getSymbol().getRootReference().getValue())) {
        if (cuf::isRegisteredDeviceGlobal(global)) {
          rewriter.setInsertionPointAfter(addrOfOp);
          auto mod = op->getParentOfType<mlir::ModuleOp>();
          fir::FirOpBuilder builder(rewriter, mod);
          mlir::Location loc = op.getLoc();
          mlir::func::FuncOp callee =
              fir::runtime::getRuntimeFunc<mkRTKey(CUFGetDeviceAddress)>(
                  loc, builder);
          auto fTy = callee.getFunctionType();
          mlir::Type toTy = fTy.getInput(0);
          mlir::Value inputArg =
              createConvertOp(rewriter, loc, toTy, addrOfOp.getResult());
          mlir::Value sourceFile =
              fir::factory::locationToFilename(builder, loc);
          mlir::Value sourceLine =
              fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
          llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
              builder, loc, fTy, inputArg, sourceFile, sourceLine)};
          auto call = rewriter.create<fir::CallOp>(loc, callee, args);
          mlir::Value cast = createConvertOp(
              rewriter, loc, op.getMemref().getType(), call->getResult(0));
          rewriter.startOpModification(op);
          op.getMemrefMutable().assign(cast);
          rewriter.finalizeOpModification(op);
          return success();
        }
      }
    }
    return failure();
  }

private:
  const mlir::SymbolTable &symTab;
};

struct CUFFreeOpConversion : public mlir::OpRewritePattern<cuf::FreeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::FreeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (inDeviceContext(op.getOperation())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (!mlir::isa<fir::ReferenceType>(op.getDevptr().getType()))
      return failure();

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);

    auto refTy = mlir::dyn_cast<fir::ReferenceType>(op.getDevptr().getType());
    if (!mlir::isa<fir::BaseBoxType>(refTy.getEleTy())) {
      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFMemFree)>(loc, builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(3));
      mlir::Value memTy = builder.createIntegerConstant(
          loc, builder.getI32Type(), getMemType(op.getDataAttr()));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, op.getDevptr(), memTy, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    // Convert cuf.free on descriptors.
    mlir::func::FuncOp func =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFFreeDescriptor)>(loc, builder);
    auto fTy = func.getFunctionType();
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, op.getDevptr(), sourceFile, sourceLine)};
    builder.create<fir::CallOp>(loc, func, args);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static bool isDstGlobal(cuf::DataTransferOp op) {
  if (auto declareOp = op.getDst().getDefiningOp<fir::DeclareOp>())
    if (declareOp.getMemref().getDefiningOp<fir::AddrOfOp>())
      return true;
  if (auto declareOp = op.getDst().getDefiningOp<hlfir::DeclareOp>())
    if (declareOp.getMemref().getDefiningOp<fir::AddrOfOp>())
      return true;
  return false;
}

static mlir::Value getShapeFromDecl(mlir::Value src) {
  if (auto declareOp = src.getDefiningOp<fir::DeclareOp>())
    return declareOp.getShape();
  if (auto declareOp = src.getDefiningOp<hlfir::DeclareOp>())
    return declareOp.getShape();
  return mlir::Value{};
}

static mlir::Value emboxSrc(mlir::PatternRewriter &rewriter,
                            cuf::DataTransferOp op,
                            const mlir::SymbolTable &symtab) {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Value addr;
  mlir::Type srcTy = fir::unwrapRefType(op.getSrc().getType());
  if (fir::isa_trivial(srcTy) &&
      mlir::matchPattern(op.getSrc().getDefiningOp(), mlir::m_Constant())) {
    mlir::Value src = op.getSrc();
    if (srcTy.isInteger(1)) {
      // i1 is not a supported type in the descriptor and it is actually coming
      // from a LOGICAL constant. Store it as a fir.logical.
      srcTy = fir::LogicalType::get(rewriter.getContext(), 4);
      src = createConvertOp(rewriter, loc, srcTy, src);
    }
    // Put constant in memory if it is not.
    mlir::Value alloc = builder.createTemporary(loc, srcTy);
    builder.create<fir::StoreOp>(loc, src, alloc);
    addr = alloc;
  } else {
    addr = op.getSrc();
  }
  llvm::SmallVector<mlir::Value> lenParams;
  mlir::Type boxTy = fir::BoxType::get(srcTy);
  mlir::Value box =
      builder.createBox(loc, boxTy, addr, getShapeFromDecl(op.getSrc()),
                        /*slice=*/nullptr, lenParams,
                        /*tdesc=*/nullptr);
  mlir::Value src = builder.createTemporary(loc, box.getType());
  builder.create<fir::StoreOp>(loc, box, src);
  return src;
}

static mlir::Value emboxDst(mlir::PatternRewriter &rewriter,
                            cuf::DataTransferOp op,
                            const mlir::SymbolTable &symtab) {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  mlir::Location loc = op.getLoc();
  fir::FirOpBuilder builder(rewriter, mod);
  mlir::Type dstTy = fir::unwrapRefType(op.getDst().getType());
  mlir::Value dstAddr = op.getDst();
  mlir::Type dstBoxTy = fir::BoxType::get(dstTy);
  llvm::SmallVector<mlir::Value> lenParams;
  mlir::Value dstBox =
      builder.createBox(loc, dstBoxTy, dstAddr, getShapeFromDecl(op.getDst()),
                        /*slice=*/nullptr, lenParams,
                        /*tdesc=*/nullptr);
  mlir::Value dst = builder.createTemporary(loc, dstBox.getType());
  builder.create<fir::StoreOp>(loc, dstBox, dst);
  return dst;
}

struct CUFDataTransferOpConversion
    : public mlir::OpRewritePattern<cuf::DataTransferOp> {
  using OpRewritePattern::OpRewritePattern;

  CUFDataTransferOpConversion(mlir::MLIRContext *context,
                              const mlir::SymbolTable &symtab,
                              mlir::DataLayout *dl,
                              const fir::LLVMTypeConverter *typeConverter)
      : OpRewritePattern(context), symtab{symtab}, dl{dl},
        typeConverter{typeConverter} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::DataTransferOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Type srcTy = fir::unwrapRefType(op.getSrc().getType());
    mlir::Type dstTy = fir::unwrapRefType(op.getDst().getType());

    mlir::Location loc = op.getLoc();
    unsigned mode = 0;
    if (op.getTransferKind() == cuf::DataTransferKind::HostDevice) {
      mode = kHostToDevice;
    } else if (op.getTransferKind() == cuf::DataTransferKind::DeviceHost) {
      mode = kDeviceToHost;
    } else if (op.getTransferKind() == cuf::DataTransferKind::DeviceDevice) {
      mode = kDeviceToDevice;
    } else {
      mlir::emitError(loc, "unsupported transfer kind\n");
    }

    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    mlir::Value modeValue =
        builder.createIntegerConstant(loc, builder.getI32Type(), mode);

    // Convert data transfer without any descriptor.
    if (!mlir::isa<fir::BaseBoxType>(srcTy) &&
        !mlir::isa<fir::BaseBoxType>(dstTy)) {

      if (fir::isa_trivial(srcTy) && !fir::isa_trivial(dstTy)) {
        // Initialization of an array from a scalar value should be implemented
        // via a kernel launch. Use the flan runtime via the Assign function
        // until we have more infrastructure.
        mlir::Value src = emboxSrc(rewriter, op, symtab);
        mlir::Value dst = emboxDst(rewriter, op, symtab);
        mlir::func::FuncOp func =
            fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferCstDesc)>(
                loc, builder);
        auto fTy = func.getFunctionType();
        mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
        mlir::Value sourceLine =
            fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
        llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
            builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
        builder.create<fir::CallOp>(loc, func, args);
        rewriter.eraseOp(op);
        return mlir::success();
      }

      mlir::Type i64Ty = builder.getI64Type();
      mlir::Value nbElement;
      if (op.getShape()) {
        llvm::SmallVector<mlir::Value> extents;
        if (auto shapeOp =
                mlir::dyn_cast<fir::ShapeOp>(op.getShape().getDefiningOp())) {
          extents = shapeOp.getExtents();
        } else if (auto shapeShiftOp = mlir::dyn_cast<fir::ShapeShiftOp>(
                       op.getShape().getDefiningOp())) {
          for (auto i : llvm::enumerate(shapeShiftOp.getPairs()))
            if (i.index() & 1)
              extents.push_back(i.value());
        }

        nbElement = rewriter.create<fir::ConvertOp>(loc, i64Ty, extents[0]);
        for (unsigned i = 1; i < extents.size(); ++i) {
          auto operand =
              rewriter.create<fir::ConvertOp>(loc, i64Ty, extents[i]);
          nbElement =
              rewriter.create<mlir::arith::MulIOp>(loc, nbElement, operand);
        }
      } else {
        if (auto seqTy = mlir::dyn_cast_or_null<fir::SequenceType>(dstTy))
          nbElement = builder.createIntegerConstant(
              loc, i64Ty, seqTy.getConstantArraySize());
      }
      unsigned width = 0;
      if (fir::isa_derived(fir::unwrapSequenceType(dstTy))) {
        mlir::Type structTy =
            typeConverter->convertType(fir::unwrapSequenceType(dstTy));
        width = dl->getTypeSizeInBits(structTy) / 8;
      } else {
        width = computeWidth(loc, dstTy, kindMap);
      }
      mlir::Value widthValue = rewriter.create<mlir::arith::ConstantOp>(
          loc, i64Ty, rewriter.getIntegerAttr(i64Ty, width));
      mlir::Value bytes =
          nbElement
              ? rewriter.create<mlir::arith::MulIOp>(loc, nbElement, widthValue)
              : widthValue;

      mlir::func::FuncOp func =
          fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferPtrPtr)>(loc,
                                                                       builder);
      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(5));

      mlir::Value dst = op.getDst();
      mlir::Value src = op.getSrc();
      // Materialize the src if constant.
      if (matchPattern(src.getDefiningOp(), mlir::m_Constant())) {
        mlir::Value temp = builder.createTemporary(loc, srcTy);
        builder.create<fir::StoreOp>(loc, src, temp);
        src = temp;
      }
      llvm::SmallVector<mlir::Value> args{
          fir::runtime::createArguments(builder, loc, fTy, dst, src, bytes,
                                        modeValue, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
      return mlir::success();
    }

    auto materializeBoxIfNeeded = [&](mlir::Value val) -> mlir::Value {
      if (mlir::isa<fir::EmboxOp, fir::ReboxOp>(val.getDefiningOp())) {
        // Materialize the box to memory to be able to call the runtime.
        mlir::Value box = builder.createTemporary(loc, val.getType());
        builder.create<fir::StoreOp>(loc, val, box);
        return box;
      }
      return val;
    };

    // Conversion of data transfer involving at least one descriptor.
    if (mlir::isa<fir::BaseBoxType>(dstTy)) {
      // Transfer to a descriptor.
      mlir::func::FuncOp func =
          isDstGlobal(op)
              ? fir::runtime::getRuntimeFunc<mkRTKey(
                    CUFDataTransferGlobalDescDesc)>(loc, builder)
              : fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferDescDesc)>(
                    loc, builder);
      mlir::Value dst = op.getDst();
      mlir::Value src = op.getSrc();
      if (!mlir::isa<fir::BaseBoxType>(srcTy)) {
        src = emboxSrc(rewriter, op, symtab);
        if (fir::isa_trivial(srcTy))
          func = fir::runtime::getRuntimeFunc<mkRTKey(CUFDataTransferCstDesc)>(
              loc, builder);
      }

      src = materializeBoxIfNeeded(src);
      dst = materializeBoxIfNeeded(dst);

      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
    } else {
      // Transfer from a descriptor.
      mlir::Value dst = emboxDst(rewriter, op, symtab);
      mlir::Value src = materializeBoxIfNeeded(op.getSrc());

      mlir::func::FuncOp func = fir::runtime::getRuntimeFunc<mkRTKey(
          CUFDataTransferDescDescNoRealloc)>(loc, builder);

      auto fTy = func.getFunctionType();
      mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
      mlir::Value sourceLine =
          fir::factory::locationToLineNo(builder, loc, fTy.getInput(4));
      llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
          builder, loc, fTy, dst, src, modeValue, sourceFile, sourceLine)};
      builder.create<fir::CallOp>(loc, func, args);
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }

private:
  const mlir::SymbolTable &symtab;
  mlir::DataLayout *dl;
  const fir::LLVMTypeConverter *typeConverter;
};

struct CUFLaunchOpConversion
    : public mlir::OpRewritePattern<cuf::KernelLaunchOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  CUFLaunchOpConversion(mlir::MLIRContext *context,
                        const mlir::SymbolTable &symTab)
      : OpRewritePattern(context), symTab{symTab} {}

  mlir::LogicalResult
  matchAndRewrite(cuf::KernelLaunchOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    auto idxTy = mlir::IndexType::get(op.getContext());
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(0));
    auto gridSizeX =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idxTy, op.getGridX());
    auto gridSizeY =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idxTy, op.getGridY());
    auto gridSizeZ =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idxTy, op.getGridZ());
    auto blockSizeX =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idxTy, op.getBlockX());
    auto blockSizeY =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idxTy, op.getBlockY());
    auto blockSizeZ =
        rewriter.create<mlir::arith::IndexCastOp>(loc, idxTy, op.getBlockZ());
    auto kernelName = mlir::SymbolRefAttr::get(
        rewriter.getStringAttr(cudaDeviceModuleName),
        {mlir::SymbolRefAttr::get(
            rewriter.getContext(),
            op.getCallee().getLeafReference().getValue())});
    mlir::Value clusterDimX, clusterDimY, clusterDimZ;
    if (auto funcOp = symTab.lookup<mlir::func::FuncOp>(
            op.getCallee().getLeafReference())) {
      if (auto clusterDimsAttr = funcOp->getAttrOfType<cuf::ClusterDimsAttr>(
              cuf::getClusterDimsAttrName())) {
        clusterDimX = rewriter.create<mlir::arith::ConstantIndexOp>(
            loc, clusterDimsAttr.getX().getInt());
        clusterDimY = rewriter.create<mlir::arith::ConstantIndexOp>(
            loc, clusterDimsAttr.getY().getInt());
        clusterDimZ = rewriter.create<mlir::arith::ConstantIndexOp>(
            loc, clusterDimsAttr.getZ().getInt());
      }
    }
    auto gpuLaunchOp = rewriter.create<mlir::gpu::LaunchFuncOp>(
        loc, kernelName, mlir::gpu::KernelDim3{gridSizeX, gridSizeY, gridSizeZ},
        mlir::gpu::KernelDim3{blockSizeX, blockSizeY, blockSizeZ}, zero,
        op.getArgs());
    if (clusterDimX && clusterDimY && clusterDimZ) {
      gpuLaunchOp.getClusterSizeXMutable().assign(clusterDimX);
      gpuLaunchOp.getClusterSizeYMutable().assign(clusterDimY);
      gpuLaunchOp.getClusterSizeZMutable().assign(clusterDimZ);
    }
    rewriter.replaceOp(op, gpuLaunchOp);
    return mlir::success();
  }

private:
  const mlir::SymbolTable &symTab;
};

struct CUFSyncDescriptorOpConversion
    : public mlir::OpRewritePattern<cuf::SyncDescriptorOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(cuf::SyncDescriptorOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    fir::FirOpBuilder builder(rewriter, mod);
    mlir::Location loc = op.getLoc();

    auto globalOp = mod.lookupSymbol<fir::GlobalOp>(op.getGlobalName());
    if (!globalOp)
      return mlir::failure();

    auto hostAddr = builder.create<fir::AddrOfOp>(
        loc, fir::ReferenceType::get(globalOp.getType()), op.getGlobalName());
    mlir::func::FuncOp callee =
        fir::runtime::getRuntimeFunc<mkRTKey(CUFSyncGlobalDescriptor)>(loc,
                                                                       builder);
    auto fTy = callee.getFunctionType();
    mlir::Value sourceFile = fir::factory::locationToFilename(builder, loc);
    mlir::Value sourceLine =
        fir::factory::locationToLineNo(builder, loc, fTy.getInput(2));
    llvm::SmallVector<mlir::Value> args{fir::runtime::createArguments(
        builder, loc, fTy, hostAddr, sourceFile, sourceLine)};
    builder.create<fir::CallOp>(loc, callee, args);
    op.erase();
    return mlir::success();
  }
};

class CUFOpConversion : public fir::impl::CUFOpConversionBase<CUFOpConversion> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    mlir::ConversionTarget target(*ctx);

    mlir::Operation *op = getOperation();
    mlir::ModuleOp module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return signalPassFailure();
    mlir::SymbolTable symtab(module);

    std::optional<mlir::DataLayout> dl =
        fir::support::getOrSetDataLayout(module, /*allowDefaultLayout=*/false);
    fir::LLVMTypeConverter typeConverter(module, /*applyTBAA=*/false,
                                         /*forceUnifiedTBAATree=*/false, *dl);
    target.addLegalDialect<fir::FIROpsDialect, mlir::arith::ArithDialect,
                           mlir::gpu::GPUDialect>();
    cuf::populateCUFToFIRConversionPatterns(typeConverter, *dl, symtab,
                                            patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }

    target.addDynamicallyLegalOp<fir::DeclareOp>([&](fir::DeclareOp op) {
      if (inDeviceContext(op))
        return true;
      if (auto addrOfOp = op.getMemref().getDefiningOp<fir::AddrOfOp>()) {
        if (auto global = symtab.lookup<fir::GlobalOp>(
                addrOfOp.getSymbol().getRootReference().getValue())) {
          if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(global.getType())))
            return true;
          if (cuf::isRegisteredDeviceGlobal(global))
            return false;
        }
      }
      return true;
    });

    patterns.clear();
    cuf::populateFIRCUFConversionPatterns(symtab, patterns);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(ctx),
                      "error in CUF op conversion\n");
      signalPassFailure();
    }
  }
};
} // namespace

void cuf::populateCUFToFIRConversionPatterns(
    const fir::LLVMTypeConverter &converter, mlir::DataLayout &dl,
    const mlir::SymbolTable &symtab, mlir::RewritePatternSet &patterns) {
  patterns.insert<CUFAllocOpConversion>(patterns.getContext(), &dl, &converter);
  patterns.insert<CUFAllocateOpConversion, CUFDeallocateOpConversion,
                  CUFFreeOpConversion, CUFSyncDescriptorOpConversion>(
      patterns.getContext());
  patterns.insert<CUFDataTransferOpConversion>(patterns.getContext(), symtab,
                                               &dl, &converter);
  patterns.insert<CUFLaunchOpConversion>(patterns.getContext(), symtab);
}

void cuf::populateFIRCUFConversionPatterns(const mlir::SymbolTable &symtab,
                                           mlir::RewritePatternSet &patterns) {
  patterns.insert<DeclareOpConversion>(patterns.getContext(), symtab);
}
