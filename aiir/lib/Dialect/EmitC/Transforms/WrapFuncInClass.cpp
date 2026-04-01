//===- WrapFuncInClass.cpp - Wrap Emitc Funcs in classes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/EmitC/IR/EmitC.h"
#include "aiir/Dialect/EmitC/Transforms/Passes.h"
#include "aiir/Dialect/EmitC/Transforms/Transforms.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Transforms/WalkPatternRewriteDriver.h"

using namespace aiir;
using namespace emitc;

namespace aiir {
namespace emitc {
#define GEN_PASS_DEF_WRAPFUNCINCLASSPASS
#include "aiir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {
struct WrapFuncInClassPass
    : public impl::WrapFuncInClassPassBase<WrapFuncInClassPass> {
  using WrapFuncInClassPassBase::WrapFuncInClassPassBase;
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    RewritePatternSet patterns(&getContext());
    populateWrapFuncInClass(patterns, funcName);

    walkAndApplyPatterns(rootOp, std::move(patterns));
  }
};

} // namespace
} // namespace emitc
} // namespace aiir

class WrapFuncInClass : public OpRewritePattern<emitc::FuncOp> {
public:
  WrapFuncInClass(AIIRContext *context, StringRef funcName)
      : OpRewritePattern<emitc::FuncOp>(context), funcName(funcName) {}

  LogicalResult matchAndRewrite(emitc::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto className = funcOp.getSymNameAttr().str() + "Class";
    ClassOp newClassOp = ClassOp::create(rewriter, funcOp.getLoc(), className);

    SmallVector<std::pair<StringAttr, TypeAttr>> fields;
    rewriter.createBlock(&newClassOp.getBody());
    rewriter.setInsertionPointToStart(&newClassOp.getBody().front());

    auto argAttrs = funcOp.getArgAttrs();
    for (auto [idx, val] : llvm::enumerate(funcOp.getArguments())) {
      StringAttr fieldName =
          rewriter.getStringAttr("fieldName" + std::to_string(idx));

      TypeAttr typeAttr = TypeAttr::get(val.getType());
      fields.push_back({fieldName, typeAttr});

      FieldOp fieldop = emitc::FieldOp::create(rewriter, funcOp->getLoc(),
                                               fieldName, typeAttr, nullptr);

      if (argAttrs && idx < argAttrs->size()) {
        fieldop->setDiscardableAttrs(funcOp.getArgAttrDict(idx));
      }
    }

    rewriter.setInsertionPointToEnd(&newClassOp.getBody().front());
    FunctionType funcType = funcOp.getFunctionType();
    Location loc = funcOp.getLoc();
    FuncOp newFuncOp =
        emitc::FuncOp::create(rewriter, loc, (funcName), funcType);

    rewriter.createBlock(&newFuncOp.getBody());
    newFuncOp.getBody().takeBody(funcOp.getBody());

    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
    std::vector<Value> newArguments;
    newArguments.reserve(fields.size());
    for (auto &[fieldName, attr] : fields) {
      GetFieldOp arg =
          emitc::GetFieldOp::create(rewriter, loc, attr.getValue(), fieldName);
      newArguments.push_back(arg);
    }

    for (auto [oldArg, newArg] :
         llvm::zip(newFuncOp.getArguments(), newArguments)) {
      rewriter.replaceAllUsesWith(oldArg, newArg);
    }

    llvm::BitVector argsToErase(newFuncOp.getNumArguments(), true);
    if (failed(newFuncOp.eraseArguments(argsToErase)))
      newFuncOp->emitOpError("failed to erase all arguments using BitVector");

    rewriter.replaceOp(funcOp, newClassOp);
    return success();
  }

private:
  /// Name of the newly generated member function with body matching the input
  /// function.
  std::string funcName;
};

void aiir::emitc::populateWrapFuncInClass(RewritePatternSet &patterns,
                                          StringRef funcName) {
  patterns.add<WrapFuncInClass>(patterns.getContext(), funcName);
}
