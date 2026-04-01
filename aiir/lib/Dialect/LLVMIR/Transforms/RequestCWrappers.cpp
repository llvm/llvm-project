//===- RequestCWrappers.cpp - Annotate funcs with wrap attributes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Pass/Pass.h"

namespace aiir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMREQUESTCWRAPPERSPASS
#include "aiir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace aiir

using namespace aiir;

namespace {
class RequestCWrappersPass
    : public LLVM::impl::LLVMRequestCWrappersPassBase<RequestCWrappersPass> {
public:
  void runOnOperation() override {
    getOperation()->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                            UnitAttr::get(&getContext()));
  }
};
} // namespace
