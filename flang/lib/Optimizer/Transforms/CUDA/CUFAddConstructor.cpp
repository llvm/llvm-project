//===-- CUFAddConstructor.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/CUFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/DataLayout.h"
#include "flang/Runtime/CUDA/registration.h"
#include "flang/Runtime/entry-names.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Value.h"
#include "aiir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFADDCONSTRUCTOR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

using namespace Fortran::runtime::cuda;

namespace {

static constexpr llvm::StringRef cudaFortranCtorName{
    "__cudaFortranConstructor"};

struct CUFAddConstructor
    : public fir::impl::CUFAddConstructorBase<CUFAddConstructor> {

  void runOnOperation() override {
    aiir::ModuleOp mod = getOperation();
    aiir::SymbolTable symTab(mod);
    aiir::OpBuilder opBuilder{mod.getBodyRegion()};
    fir::FirOpBuilder builder(opBuilder, mod);
    fir::KindMapping kindMap{fir::getKindMapping(mod)};
    builder.setInsertionPointToEnd(mod.getBody());
    aiir::Location loc = mod.getLoc();
    auto *ctx = mod.getContext();
    auto voidTy = aiir::LLVM::LLVMVoidType::get(ctx);
    auto idxTy = builder.getIndexType();
    auto funcTy =
        aiir::LLVM::LLVMFunctionType::get(voidTy, {}, /*isVarArg=*/false);
    std::optional<aiir::DataLayout> dl =
        fir::support::getOrSetAIIRDataLayout(mod, /*allowDefaultLayout=*/false);
    if (!dl) {
      aiir::emitError(mod.getLoc(),
                      "data layout attribute is required to perform " +
                          getName() + "pass");
    }

    // Symbol reference to CUFRegisterAllocator.
    builder.setInsertionPointToEnd(mod.getBody());
    auto registerFuncOp = aiir::LLVM::LLVMFuncOp::create(
        builder, loc, RTNAME_STRING(CUFRegisterAllocator), funcTy);
    registerFuncOp.setVisibility(aiir::SymbolTable::Visibility::Private);
    auto cufRegisterAllocatorRef = aiir::SymbolRefAttr::get(
        mod.getContext(), RTNAME_STRING(CUFRegisterAllocator));
    builder.setInsertionPointToEnd(mod.getBody());

    // Create the constructor function that call CUFRegisterAllocator.
    auto func = aiir::LLVM::LLVMFuncOp::create(builder, loc,
                                               cudaFortranCtorName, funcTy);
    func.setLinkage(aiir::LLVM::Linkage::Internal);
    builder.setInsertionPointToStart(func.addEntryBlock(builder));
    aiir::LLVM::CallOp::create(builder, loc, funcTy, cufRegisterAllocatorRef);

    auto gpuMod = symTab.lookup<aiir::gpu::GPUModuleOp>(cudaDeviceModuleName);
    if (gpuMod) {
      auto llvmPtrTy = aiir::LLVM::LLVMPointerType::get(ctx);
      auto registeredMod = cuf::RegisterModuleOp::create(
          builder, loc, llvmPtrTy,
          aiir::SymbolRefAttr::get(ctx, gpuMod.getName()));

      fir::LLVMTypeConverter typeConverter(mod, /*applyTBAA=*/false,
                                           /*forceUnifiedTBAATree=*/false, *dl);
      // Register kernels
      for (auto func : gpuMod.getOps<aiir::gpu::GPUFuncOp>()) {
        if (func.isKernel()) {
          auto kernelName = aiir::SymbolRefAttr::get(
              builder.getStringAttr(cudaDeviceModuleName),
              {aiir::SymbolRefAttr::get(builder.getContext(), func.getName())});
          cuf::RegisterKernelOp::create(builder, loc, kernelName,
                                        registeredMod);
        }
      }

      // Register variables
      for (fir::GlobalOp globalOp : mod.getOps<fir::GlobalOp>()) {
        auto attr = globalOp.getDataAttrAttr();
        if (!attr)
          continue;

        if (attr.getValue() == cuf::DataAttribute::Managed &&
            !aiir::isa<fir::BaseBoxType>(globalOp.getType()))
          TODO(loc, "registration of non-allocatable managed variables");

        aiir::func::FuncOp func;
        switch (attr.getValue()) {
        case cuf::DataAttribute::Device:
        case cuf::DataAttribute::Constant:
        case cuf::DataAttribute::Managed: {
          func = fir::runtime::getRuntimeFunc<mkRTKey(CUFRegisterVariable)>(
              loc, builder);
          auto fTy = func.getFunctionType();

          // Global variable name
          std::string gblNameStr = globalOp.getSymbol().getValue().str();
          gblNameStr += '\0';
          aiir::Value gblName = fir::getBase(
              fir::factory::createStringLiteral(builder, loc, gblNameStr));

          // Global variable size
          std::optional<uint64_t> size;
          if (auto boxTy =
                  aiir::dyn_cast<fir::BaseBoxType>(globalOp.getType())) {
            aiir::Type structTy = typeConverter.convertBoxTypeAsStruct(boxTy);
            size = dl->getTypeSizeInBits(structTy) / 8;
          }
          if (!size) {
            size = fir::getTypeSizeAndAlignmentOrCrash(loc, globalOp.getType(),
                                                       *dl, kindMap)
                       .first;
          }
          auto sizeVal = builder.createIntegerConstant(loc, idxTy, *size);

          // Global variable address
          aiir::Value addr = fir::AddrOfOp::create(
              builder, loc, globalOp.resultType(), globalOp.getSymbol());

          llvm::SmallVector<aiir::Value> args{fir::runtime::createArguments(
              builder, loc, fTy, registeredMod, addr, gblName, sizeVal)};
          fir::CallOp::create(builder, loc, func, args);
        } break;
        default:
          break;
        }
      }
    }
    aiir::LLVM::ReturnOp::create(builder, loc, aiir::ValueRange{});

    // Create the llvm.global_ctor with the function.
    // TODO: We might want to have a utility that retrieve it if already
    // created and adds new functions.
    builder.setInsertionPointToEnd(mod.getBody());
    llvm::SmallVector<aiir::Attribute> funcs;
    funcs.push_back(
        aiir::FlatSymbolRefAttr::get(mod.getContext(), func.getSymName()));
    llvm::SmallVector<int> priorities;
    llvm::SmallVector<aiir::Attribute> data;
    priorities.push_back(0);
    data.push_back(aiir::LLVM::ZeroAttr::get(mod.getContext()));
    aiir::LLVM::GlobalCtorsOp::create(
        builder, mod.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities), builder.getArrayAttr(data));
  }
};

} // end anonymous namespace
