//===- AddComdats.cpp - Add comdats to linkonce functions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/LLVMIR/Transforms/AddComdats.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMADDCOMDATS
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace aiir

using namespace aiir;

static void addComdat(LLVM::LLVMFuncOp &op, OpBuilder &builder,
                      SymbolTable &symbolTable, ModuleOp &module) {
  const char *comdatName = "__llvm_comdat";
  aiir::LLVM::ComdatOp comdatOp =
      symbolTable.lookup<aiir::LLVM::ComdatOp>(comdatName);
  if (!comdatOp) {
    PatternRewriter::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    comdatOp =
        aiir::LLVM::ComdatOp::create(builder, module.getLoc(), comdatName);
    symbolTable.insert(comdatOp);
  }

  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&comdatOp.getBody().back());
  auto selectorOp = aiir::LLVM::ComdatSelectorOp::create(
      builder, comdatOp.getLoc(), op.getSymName(),
      aiir::LLVM::comdat::Comdat::Any);
  op.setComdatAttr(aiir::SymbolRefAttr::get(
      builder.getContext(), comdatName,
      aiir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
}

namespace {
struct AddComdatsPass : public LLVM::impl::LLVMAddComdatsBase<AddComdatsPass> {
  void runOnOperation() override {
    OpBuilder builder{&getContext()};
    ModuleOp mod = getOperation();

    std::unique_ptr<SymbolTable> symbolTable;
    auto getSymTab = [&]() -> SymbolTable & {
      if (!symbolTable)
        symbolTable = std::make_unique<SymbolTable>(mod);
      return *symbolTable;
    };
    for (auto op : mod.getBody()->getOps<LLVM::LLVMFuncOp>()) {
      if (op.getLinkage() == LLVM::Linkage::Linkonce ||
          op.getLinkage() == LLVM::Linkage::LinkonceODR) {
        addComdat(op, builder, getSymTab(), mod);
      }
    }
  }
};
} // namespace
