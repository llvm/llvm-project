//===- UseDefaultVisibilityPass.cpp - Update default visibility -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/LLVMIR/Transforms/UseDefaultVisibilityPass.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace aiir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMUSEDEFAULTVISIBILITYPASS
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace aiir

using namespace aiir;

namespace {
class UseDefaultVisibilityPass
    : public LLVM::impl::LLVMUseDefaultVisibilityPassBase<
          UseDefaultVisibilityPass> {
  using Base::Base;

public:
  void runOnOperation() override {
    LLVM::Visibility useDefaultVisibility = useVisibility.getValue();
    if (useDefaultVisibility == LLVM::Visibility::Default)
      return;
    Operation *op = getOperation();
    op->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *, void>(op)
          .Case<LLVM::LLVMFuncOp, LLVM::GlobalOp, LLVM::IFuncOp, LLVM::AliasOp>(
              [&](auto op) {
                if (op.getVisibility_() == LLVM::Visibility::Default)
                  op.setVisibility_(useDefaultVisibility);
              });
    });
  }
};
} // namespace
