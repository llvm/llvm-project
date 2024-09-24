//===- CUFAddConstructor.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace fir {
#define GEN_PASS_DEF_CUFADDCONSTRUCTOR
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {

static constexpr llvm::StringRef cudaFortranCtorName{
    "__cudaFortranConstructor"};

struct CUFAddConstructor
    : public fir::impl::CUFAddConstructorBase<CUFAddConstructor> {

  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mlir::OpBuilder builder{mod.getBodyRegion()};
    builder.setInsertionPointToEnd(mod.getBody());
    mlir::Location loc = mod.getLoc();
    auto *ctx = mod.getContext();
    auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
    auto funcTy =
        mlir::LLVM::LLVMFunctionType::get(voidTy, {}, /*isVarArg=*/false);

    llvm::errs() << "DBG1\n";
    // Symbol reference to CUFRegisterAllocator.
    builder.setInsertionPointToEnd(mod.getBody());
    llvm::errs() << "DBG1.1\n";
    auto registerFuncOp = builder.create<mlir::LLVM::LLVMFuncOp>(
        loc, RTNAME_STRING(CUFRegisterAllocator), funcTy);
    llvm::errs() << "DBG1.2\n";
    registerFuncOp.setVisibility(mlir::SymbolTable::Visibility::Private);
    auto cufRegisterAllocatorRef = mlir::SymbolRefAttr::get(
        mod.getContext(), RTNAME_STRING(CUFRegisterAllocator));
    llvm::errs() << "DBG1.3\n";
    builder.setInsertionPointToEnd(mod.getBody());
    // mod.push_back(registerFuncOp);
    llvm::errs() << "DBG1.4\n";

    llvm::errs() << "DBG2\n";
    // Create the constructor function that cal CUFRegisterAllocator.
    builder.setInsertionPointToEnd(mod.getBody());
    auto func = builder.create<mlir::LLVM::LLVMFuncOp>(loc, cudaFortranCtorName,
                                                       funcTy);
    func.setLinkage(mlir::LLVM::Linkage::Internal);
    builder.setInsertionPointToStart(func.addEntryBlock(builder));
    builder.create<mlir::LLVM::CallOp>(loc, funcTy, cufRegisterAllocatorRef);
    builder.create<mlir::LLVM::ReturnOp>(loc, mlir::ValueRange{});
    // mod.push_back(func);

    llvm::errs() << "DBG3\n";
    // Create the llvm.global_ctor with the function.
    // TODO: We might want to have a utility that retrieve it if already created
    // and adds new functions.
    builder.setInsertionPointToEnd(mod.getBody());
    llvm::SmallVector<mlir::Attribute> funcs;
    funcs.push_back(
        mlir::FlatSymbolRefAttr::get(mod.getContext(), func.getSymName()));
    llvm::SmallVector<int> priorities;
    priorities.push_back(0);
    builder.create<mlir::LLVM::GlobalCtorsOp>(
        mod.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities));
  }
};

} // end anonymous namespace