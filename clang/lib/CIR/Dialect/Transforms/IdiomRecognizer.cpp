//===- IdiomRecognizer.cpp - Recognize and raise C/C++ library calls ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/Module.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"

using cir::CIRBaseBuilderTy;
using namespace mlir;
using namespace mlir::cir;

namespace {

struct IdiomRecognizerPass : public IdiomRecognizerBase<IdiomRecognizerPass> {
  IdiomRecognizerPass() = default;
  void runOnOperation() override;
  void recognizeCall(CallOp call);
  void raiseStdFind(CallOp call);
  void raiseIteratorBeginEnd(CallOp call);

  // Handle pass options
  struct Options {
    enum : unsigned {
      None = 0,
      RemarkFoundCalls = 1,
      RemarkAll = 1 << 1,
    };
    unsigned val = None;
    bool isOptionsParsed = false;

    void parseOptions(ArrayRef<StringRef> remarks) {
      if (isOptionsParsed)
        return;

      for (auto &remark : remarks) {
        val |= StringSwitch<unsigned>(remark)
                   .Case("found-calls", RemarkFoundCalls)
                   .Case("all", RemarkAll)
                   .Default(None);
      }
      isOptionsParsed = true;
    }

    void parseOptions(IdiomRecognizerPass &pass) {
      SmallVector<llvm::StringRef, 4> remarks;

      for (auto &r : pass.remarksList)
        remarks.push_back(r);

      parseOptions(remarks);
    }

    bool emitRemarkAll() { return val & RemarkAll; }
    bool emitRemarkFoundCalls() {
      return emitRemarkAll() || val & RemarkFoundCalls;
    }
  } opts;

  ///
  /// AST related
  /// -----------
  clang::ASTContext *astCtx;
  void setASTContext(clang::ASTContext *c) { astCtx = c; }

  /// Tracks current module.
  ModuleOp theModule;
};
} // namespace

void IdiomRecognizerPass::raiseStdFind(CallOp call) {
  // FIXME: tablegen all of this function.
  if (call.getNumOperands() != 3)
    return;

  auto callExprAttr = call.getAstAttr();
  if (!callExprAttr || !callExprAttr.isStdFunctionCall("find")) {
    return;
  }

  if (opts.emitRemarkFoundCalls())
    emitRemark(call.getLoc()) << "found call to std::find()";

  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointAfter(call.getOperation());
  auto findOp = builder.create<mlir::cir::StdFindOp>(
      call.getLoc(), call.getResult(0).getType(), call.getCalleeAttr(),
      call.getOperand(0), call.getOperand(1), call.getOperand(2));

  call.replaceAllUsesWith(findOp);
  call.erase();
}

static bool isIteratorLikeType(mlir::Type t) {
  // TODO: some iterators are going to be represented with structs,
  // in which case we could look at ASTRecordDeclInterface for more
  // information.
  auto pTy = t.dyn_cast<PointerType>();
  if (!pTy || !pTy.getPointee().isa<mlir::cir::IntType>())
    return false;
  return true;
}

static bool isIteratorInStdContainter(mlir::Type t) {
  auto sTy = t.dyn_cast<StructType>();
  if (!sTy)
    return false;
  auto recordDecl = sTy.getAst();
  if (!recordDecl.isInStdNamespace())
    return false;

  // TODO: only std::array supported for now, generalize and
  // use tablegen. CallDescription.cpp in the static analyzer
  // could be a good inspiration source too.
  if (recordDecl.getName().compare("array") != 0)
    return false;

  return true;
}

void IdiomRecognizerPass::raiseIteratorBeginEnd(CallOp call) {
  // FIXME: tablegen all of this function.
  CIRBaseBuilderTy builder(getContext());

  if (call.getNumOperands() != 1 || call.getNumResults() != 1)
    return;

  auto callExprAttr = call.getAstAttr();
  if (!callExprAttr)
    return;

  if (!isIteratorLikeType(call.getResult(0).getType()))
    return;

  // First argument is the container "this" pointer.
  auto thisPtr = call.getOperand(0).getType().dyn_cast<PointerType>();
  if (!thisPtr || !isIteratorInStdContainter(thisPtr.getPointee()))
    return;

  builder.setInsertionPointAfter(call.getOperation());
  mlir::Operation *iterOp;
  if (callExprAttr.isIteratorBeginCall()) {
    if (opts.emitRemarkFoundCalls())
      emitRemark(call.getLoc()) << "found call to begin() iterator";
    iterOp = builder.create<mlir::cir::IterBeginOp>(
        call.getLoc(), call.getResult(0).getType(), call.getCalleeAttr(),
        call.getOperand(0));
  } else if (callExprAttr.isIteratorEndCall()) {
    if (opts.emitRemarkFoundCalls())
      emitRemark(call.getLoc()) << "found call to end() iterator";
    iterOp = builder.create<mlir::cir::IterEndOp>(
        call.getLoc(), call.getResult(0).getType(), call.getCalleeAttr(),
        call.getOperand(0));
  } else {
    return;
  }

  call.replaceAllUsesWith(iterOp);
  call.erase();
}

void IdiomRecognizerPass::recognizeCall(CallOp call) {
  raiseIteratorBeginEnd(call);
  raiseStdFind(call);
}

void IdiomRecognizerPass::runOnOperation() {
  assert(astCtx && "Missing ASTContext, please construct with the right ctor");
  opts.parseOptions(*this);
  auto *op = getOperation();
  if (isa<::mlir::ModuleOp>(op))
    theModule = cast<::mlir::ModuleOp>(op);

  SmallVector<CallOp> callsToTransform;
  op->walk([&](CallOp callOp) {
    // Process call operations

    // Skip indirect calls.
    auto c = callOp.getCallee();
    if (!c)
      return;
    callsToTransform.push_back(callOp);
  });

  for (auto c : callsToTransform)
    recognizeCall(c);
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}

std::unique_ptr<Pass>
mlir::createIdiomRecognizerPass(clang::ASTContext *astCtx) {
  auto pass = std::make_unique<IdiomRecognizerPass>();
  pass->setASTContext(astCtx);
  return std::move(pass);
}
