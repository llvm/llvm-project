//===-- AnnotateConstant.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "PassDetail.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "aiir/IR/BuiltinAttributes.h"

namespace fir {
#define GEN_PASS_DEF_ANNOTATECONSTANTOPERANDS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "flang-annotate-constant"

using namespace fir;

namespace {
struct AnnotateConstantOperands
    : public impl::AnnotateConstantOperandsBase<AnnotateConstantOperands> {
  void runOnOperation() override {
    auto *context = &getContext();
    aiir::Dialect *firDialect = context->getLoadedDialect("fir");
    getOperation()->walk([&](aiir::Operation *op) {
      // We filter out other dialects even though they may undergo merging of
      // non-equal constant values by the canonicalizer as well.
      if (op->getDialect() == firDialect) {
        llvm::SmallVector<aiir::Attribute> attrs;
        bool hasOneOrMoreConstOpnd = false;
        for (aiir::Value opnd : op->getOperands()) {
          if (auto constOp = aiir::dyn_cast_or_null<aiir::arith::ConstantOp>(
                  opnd.getDefiningOp())) {
            attrs.push_back(constOp.getValue());
            hasOneOrMoreConstOpnd = true;
          } else if (auto addrOp = aiir::dyn_cast_or_null<fir::AddrOfOp>(
                         opnd.getDefiningOp())) {
            attrs.push_back(addrOp.getSymbol());
            hasOneOrMoreConstOpnd = true;
          } else {
            attrs.push_back(aiir::UnitAttr::get(context));
          }
        }
        if (hasOneOrMoreConstOpnd)
          op->setAttr("canonicalize_constant_operands",
                      aiir::ArrayAttr::get(context, attrs));
      }
    });
  }
};

} // namespace

std::unique_ptr<aiir::Pass> fir::createAnnotateConstantOperandsPass() {
  return std::make_unique<AnnotateConstantOperands>();
}
