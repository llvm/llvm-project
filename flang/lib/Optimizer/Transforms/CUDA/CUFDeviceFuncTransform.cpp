//===-- CUFDeviceFuncTransform.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
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

  static gpu::GPUFuncOp createGPUFuncOp(mlir::func::FuncOp funcOp,
                                        bool isGlobal, int computeCap) {
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

    auto deviceFuncOp =
        gpu::GPUFuncOp::create(builder, loc, funcOp.getName(), type,
                               mlir::TypeRange{}, mlir::TypeRange{});
    if (isGlobal)
      deviceFuncOp->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                            builder.getUnitAttr());

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
    mlir::OpBuilder modBuilder(mod.getBodyRegion());
    modBuilder.setInsertionPointToEnd(mod.getBody());
    auto emptyStub = func::FuncOp::create(modBuilder, loc, funcOp.getName(),
                                          funcOp.getFunctionType());
    emptyStub.setVisibility(funcOp.getVisibility());
    emptyStub->setAttrs(funcOp->getAttrs());
    auto entryBlock = emptyStub.addEntryBlock();
    modBuilder.setInsertionPointToEnd(entryBlock);
    func::ReturnOp::create(modBuilder, loc);

    symTab.erase(funcOp);
    symTab.insert(emptyStub);
  }

  static bool isDeviceFunc(mlir::func::FuncOp funcOp) {
    if (auto cudaProcAttr =
            funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
                cuf::getProcAttrName()))
      if (cudaProcAttr.getValue() == cuf::ProcAttribute::Device ||
          cudaProcAttr.getValue() == cuf::ProcAttribute::Global ||
          cudaProcAttr.getValue() == cuf::ProcAttribute::GridGlobal ||
          cudaProcAttr.getValue() == cuf::ProcAttribute::HostDevice)
        return true;
    return false;
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

    llvm::SetVector<mlir::func::FuncOp> funcsToClone;
    llvm::SetVector<mlir::func::FuncOp> deviceFuncs;
    llvm::SetVector<mlir::func::FuncOp> keepInModule;
    llvm::StringSet<> deviceFuncNames;

    // Look for all function to migrate to the GPU module.
    mod.walk([&](mlir::func::FuncOp op) {
      if (isDeviceFunc(op)) {
        deviceFuncs.insert(op);
        deviceFuncNames.insert(op.getSymName());
      }
    });

    auto processCallOp = [&](fir::CallOp op) {
      if (op.getCallee()) {
        auto func = symbolTable.lookup<mlir::func::FuncOp>(
            op.getCallee()->getLeafReference());
        if (deviceFuncs.count(func) == 0)
          funcsToClone.insert(func);
      }
    };

    // Gather all function called by device functions.
    for (auto funcOp : deviceFuncs) {
      funcOp.walk([&](fir::CallOp op) { processCallOp(op); });
      funcOp.walk([&](fir::DispatchOp op) {
        TODO(op.getLoc(), "type-bound procedure call with dynamic dispatch "
                          "in device procedure");
      });
    }

    // Functions that are referenced in a derived-type binding table must be
    // kept in the host module to avoid LLVM dialect verification errors.
    for (auto globalOp : mod.getOps<fir::GlobalOp>()) {
      if (globalOp.getName().contains(fir::kBindingTableSeparator)) {
        globalOp.walk([&](fir::AddrOfOp addrOfOp) {
          if (deviceFuncNames.contains(addrOfOp.getSymbol().getLeafReference()))
            keepInModule.insert(
                *llvm::find_if(deviceFuncs, [&](mlir::func::FuncOp f) {
                  return f.getSymName() ==
                         addrOfOp.getSymbol().getLeafReference();
                }));
        });
      }
    }

    // Gather all functions called by CUF kernels.
    mod.walk([&](cuf::KernelOp kernelOp) {
      kernelOp.walk([&](fir::CallOp op) { processCallOp(op); });
      kernelOp.walk([&](fir::DispatchOp op) {
        TODO(op.getLoc(),
             "type-bound procedure call with dynamic dispatch in cuf kernel");
      });
    });

    for (auto funcOp : funcsToClone)
      gpuModSymTab.insert(funcOp->clone());

    for (auto funcOp : deviceFuncs) {
      auto cudaProcAttr =
          funcOp.getOperation()->getAttrOfType<cuf::ProcAttributeAttr>(
              cuf::getProcAttrName());
      auto isGlobal = cudaProcAttr.getValue() == cuf::ProcAttribute::Global ||
                      cudaProcAttr.getValue() == cuf::ProcAttribute::GridGlobal;
      if (funcOp.isDeclaration()) {
        mlir::Operation *clonedFuncOp = funcOp->clone();
        if (isGlobal) {
          clonedFuncOp->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                                builder.getUnitAttr());
          clonedFuncOp->removeAttr(cuf::getProcAttrName());
          if (auto funcOp = mlir::dyn_cast<func::FuncOp>(clonedFuncOp))
            funcOp.setNested();
        }
        gpuModSymTab.insert(clonedFuncOp);
      } else {
        gpu::GPUFuncOp deviceFuncOp =
            createGPUFuncOp(funcOp, isGlobal, computeCap);
        gpuModSymTab.insert(deviceFuncOp);

        if (cudaProcAttr.getValue() != cuf::ProcAttribute::HostDevice) {
          // If the function is a global, we need to keep the host side
          // declaration for the kernel registration. Currently we just
          // erase its body but in the future, the body should be rewritten
          // to be able to launch CUDA Fortran kernel from C code.
          if (isGlobal || keepInModule.contains(funcOp))
            createHostStub(funcOp, symbolTable, mod);
          else
            funcOp.erase();
        }
      }
    }
  }
};

} // end anonymous namespace
