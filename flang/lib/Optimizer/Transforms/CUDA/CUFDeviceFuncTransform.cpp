//===-- CUFDeviceFuncTransform.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"

namespace fir {
#define GEN_PASS_DEF_CUFDEVICEFUNCTRANSFORM
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace mlir;

namespace {

class CUFDeviceFuncTransform
    : public fir::impl::CUFDeviceFuncTransformBase<CUFDeviceFuncTransform> {
  using CUFDeviceFuncTransformBase<
      CUFDeviceFuncTransform>::CUFDeviceFuncTransformBase;

  static bool isPointerLikeKernelArg(mlir::Type argType) {
    return mlir::isa<fir::ReferenceType, fir::BaseBoxType>(argType);
  }

  // Decorate INTENT(IN) kernel arguments like C "const __restrict__". NVPTX
  // only tags loads as invariant (and lowers them to ld.global.nc) when a
  // kernel pointer parameter is both readonly and noalias; see
  // NVPTXTagInvariantLoads.
  static void setIntentInKernelArgAttrs(mlir::func::FuncOp funcOp,
                                        gpu::GPUFuncOp deviceFuncOp) {
    mlir::UnitAttr unitAttr = mlir::UnitAttr::get(funcOp.getContext());

    auto markArg = [&](unsigned argIndex) {
      if (argIndex >= deviceFuncOp.getNumArguments())
        return;
      mlir::Type argType = deviceFuncOp.getArgumentTypes()[argIndex];
      if (!isPointerLikeKernelArg(argType))
        return;
      deviceFuncOp.setArgAttr(
          argIndex, mlir::LLVM::LLVMDialect::getReadonlyAttrName(), unitAttr);
      deviceFuncOp.setArgAttr(
          argIndex, mlir::LLVM::LLVMDialect::getNoAliasAttrName(), unitAttr);
    };

    funcOp.walk([&](fir::DeclareOp declareOp) {
      auto var =
          mlir::cast<fir::FortranVariableOpInterface>(declareOp.getOperation());
      if (!var.isIntentIn())
        return;
      if (auto attrs = var.getFortranAttrs())
        if (fir::bitEnumContainsAny(*attrs,
                                    fir::FortranVariableFlagsEnum::value))
          return;
      if (std::optional<uint32_t> dummyArgNo = declareOp.getDummyArgNo()) {
        // Dummy argument numbers are 1-based in FIR.
        markArg(*dummyArgNo - 1);
        return;
      }
      if (auto blockArg =
              mlir::dyn_cast<mlir::BlockArgument>(declareOp.getMemref()))
        if (blockArg.getOwner()->isEntryBlock() &&
            blockArg.getOwner()->getParentOp() == funcOp)
          markArg(blockArg.getArgNumber());
    });
  }

  static gpu::GPUFuncOp createGPUFuncOp(mlir::func::FuncOp funcOp,
                                        llvm::StringRef name, bool isGlobal,
                                        int computeCap) {
    mlir::OpBuilder builder(funcOp.getContext());

    mlir::Region &funcOpBody = funcOp.getBody();
    SetVector<Value> operands;
    for (mlir::Value operand : funcOp.getArguments())
      operands.insert(operand);

    llvm::SmallVector<mlir::Type> funcOperandTypes;
    llvm::SmallVector<mlir::Type> funcResultTypes;
    funcOperandTypes.reserve(funcOp.getArgumentTypes().size());
    funcResultTypes.reserve(funcOp.getResultTypes().size());
    for (mlir::Type opTy : funcOp.getArgumentTypes())
      funcOperandTypes.push_back(opTy);
    for (mlir::Type resTy : funcOp.getResultTypes())
      funcResultTypes.push_back(resTy);

    mlir::Location loc = funcOp.getLoc();

    mlir::FunctionType type = mlir::FunctionType::get(
        funcOp.getContext(), funcOperandTypes, funcResultTypes);

    auto deviceFuncOp = gpu::GPUFuncOp::create(
        builder, loc, name, type, mlir::TypeRange{}, mlir::TypeRange{});
    if (mlir::ArrayAttr argAttrs = funcOp.getAllArgAttrs())
      deviceFuncOp.setAllArgAttrs(argAttrs);
    setIntentInKernelArgAttrs(funcOp, deviceFuncOp);
    if (isGlobal)
      deviceFuncOp.setKernel(true);

    mlir::Region &deviceFuncBody = deviceFuncOp.getBody();
    mlir::Block &entryBlock = deviceFuncBody.front();

    mlir::IRMapping map;
    for (const auto &operand : enumerate(operands))
      map.map(operand.value(), entryBlock.getArgument(operand.index()));

    funcOpBody.cloneInto(&deviceFuncBody, map);

    deviceFuncOp.walk([](func::ReturnOp op) {
      mlir::OpBuilder replacer(op);
      gpu::ReturnOp gpuReturnOp = gpu::ReturnOp::create(replacer, op.getLoc());
      gpuReturnOp->setOperands(op.getOperands());
      op.erase();
    });

    mlir::Block &funcOpEntry = funcOp.front();
    mlir::Block *clonedFuncOpEntry = map.lookup(&funcOpEntry);

    entryBlock.getOperations().splice(entryBlock.getOperations().end(),
                                      clonedFuncOpEntry->getOperations());
    clonedFuncOpEntry->erase();

    auto launchBoundsAttr =
        funcOp.getOperation()->getAttrOfType<cuf::LaunchBoundsAttr>(
            cuf::getLaunchBoundsAttrName());
    if (launchBoundsAttr) {
      auto maxTPB = launchBoundsAttr.getMaxTPB().getInt();
      auto maxntid =
          builder.getDenseI32ArrayAttr({static_cast<int32_t>(maxTPB), 1, 1});
      deviceFuncOp->setAttr(NVVM::NVVMDialect::getMaxntidAttrName(), maxntid);
      // The minimum-blocks-per-multiprocessor operand is optional.
      if (launchBoundsAttr.getMinBPM())
        deviceFuncOp->setAttr(NVVM::NVVMDialect::getMinctasmAttrName(),
                              launchBoundsAttr.getMinBPM());
      if (computeCap >= 90 && launchBoundsAttr.getUpperBoundClusterSize())
        deviceFuncOp->setAttr(NVVM::NVVMDialect::getClusterMaxBlocksAttrName(),
                              launchBoundsAttr.getUpperBoundClusterSize());
    }

    return deviceFuncOp;
  }

  static void createHostStub(mlir::func::FuncOp funcOp,
                             mlir::SymbolTable &symTab, mlir::ModuleOp mod) {
    mlir::Location loc = funcOp.getLoc();
    // Host stub's line table needs to span the procedure body.
    mlir::Location endLoc = loc;
    if (!funcOp.getBody().empty())
      if (mlir::Operation *terminator = funcOp.getBody().back().getTerminator())
        endLoc = terminator->getLoc();
    mlir::OpBuilder modBuilder(mod.getBodyRegion());
    modBuilder.setInsertionPointToEnd(mod.getBody());
    auto emptyStub = func::FuncOp::create(modBuilder, loc, funcOp.getName(),
                                          funcOp.getFunctionType());
    emptyStub.setVisibility(funcOp.getVisibility());
    emptyStub->copyProperties(funcOp->getPropertiesStorage());
    emptyStub->setDiscardableAttrs(funcOp->getDiscardableAttrDictionary());
    auto entryBlock = emptyStub.addEntryBlock();
    modBuilder.setInsertionPointToEnd(entryBlock);
    // Add a return operation at the end of the stub with the location of the
    // original procedure's terminator.
    func::ReturnOp::create(modBuilder, endLoc);

    symTab.erase(funcOp);
    symTab.insert(emptyStub);
  }

  void runOnOperation() override {
    // Working on Module operation because inserting/removing function from the
    // module is not thread-safe.
    ModuleOp mod = getOperation();
    mlir::SymbolTable symbolTable(getOperation());

    auto *ctx = getOperation().getContext();
    mlir::OpBuilder builder(ctx);

    gpu::GPUModuleOp gpuMod = cuf::getOrCreateGPUModule(mod, symbolTable);
    mlir::SymbolTable gpuModSymTab(gpuMod);

    cuf::DeviceCodeSet code = cuf::collectDeviceCode(
        mod, symbolTable, /*rejectDynamicDispatch=*/true);

    // Optionally report an error when device code calls the runtime function
    // _FortranAioOutputDescriptor, which is not supported on the device.
    if (checkioOutputDescriptor) {
      constexpr llvm::StringRef aioOutputDescriptor =
          "_FortranAioOutputDescriptor";
      auto checkForAioOutputDescriptor = [&](fir::CallOp op) {
        if (op.getCallee() && op.getCallee()->getLeafReference().getValue() ==
                                  aioOutputDescriptor) {
          op.emitError("descriptor I/O is not supported in device code");
          signalPassFailure();
        }
      };
      for (auto funcOp : code.deviceFuncs)
        funcOp.walk(checkForAioOutputDescriptor);
      mod.walk([&](cuf::KernelOp kernelOp) {
        kernelOp.walk(checkForAioOutputDescriptor);
      });
    }

    // Device copies made by cuf-duplicate-device-func carry the symbol of the
    // procedure they copy. They take that name back on the device, where
    // cross-unit references resolve by it, and their originals stay host code.
    llvm::DenseMap<mlir::StringAttr, mlir::FlatSymbolRefAttr> originalOf;
    llvm::StringSet<> hasDeviceCopy;
    for (mlir::func::FuncOp funcOp : code.deviceFuncs)
      if (std::optional<llvm::StringRef> original =
              cuf::getDeviceCopyOf(funcOp)) {
        originalOf[funcOp.getSymNameAttr()] =
            mlir::FlatSymbolRefAttr::get(ctx, *original);
        hasDeviceCopy.insert(*original);
      }
    // cuf.kernel regions stay in host code until they are lowered, and are
    // matched to the device by the original names.
    mod.walk([&](cuf::KernelOp kernelOp) {
      cuf::remapProcedureSymbols(kernelOp, originalOf);
    });

    for (auto funcOp : code.calledFromDevice)
      if (!hasDeviceCopy.contains(funcOp.getSymName()))
        gpuModSymTab.insert(funcOp->clone());

    for (auto funcOp : code.deviceFuncs) {
      auto cudaProcAttr =
          funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
              cuf::getProcAttrName());
      auto isGlobal = cudaProcAttr.getValue() == cuf::ProcAttribute::Global ||
                      cudaProcAttr.getValue() == cuf::ProcAttribute::GridGlobal;
      std::optional<llvm::StringRef> copyOf = cuf::getDeviceCopyOf(funcOp);
      // A host_device original whose device copy exists stays host code.
      if (!copyOf && hasDeviceCopy.contains(funcOp.getSymName()))
        continue;
      llvm::StringRef deviceName = copyOf ? *copyOf : funcOp.getSymName();
      if (funcOp.isDeclaration()) {
        auto clonedFuncOp = mlir::cast<func::FuncOp>(funcOp->clone());
        if (copyOf) {
          clonedFuncOp.setSymName(deviceName);
          clonedFuncOp->removeAttr(cuf::getDeviceCopyOfAttrName());
        }
        if (isGlobal) {
          clonedFuncOp->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                                builder.getUnitAttr());
          clonedFuncOp->removeAttr(cuf::getProcAttrName());
          clonedFuncOp.setNested();
        }
        gpuModSymTab.insert(clonedFuncOp);
        if (copyOf)
          funcOp.erase();
      } else {
        gpu::GPUFuncOp deviceFuncOp =
            createGPUFuncOp(funcOp, deviceName, isGlobal, computeCap);
        cuf::remapProcedureSymbols(deviceFuncOp, originalOf);
        gpuModSymTab.insert(deviceFuncOp);

        if (cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice) {
          // If the function is a global, we need to keep the host side
          // declaration for the kernel registration. Currently we just
          // erase its body but in the future, the body should be rewritten
          // to be able to launch CUDA Fortran kernel from C code.
          if (isGlobal || code.keepInModule.contains(funcOp))
            createHostStub(funcOp, symbolTable, mod);
          else
            funcOp.erase();
        }
      }
    }
  }
};

} // end anonymous namespace
