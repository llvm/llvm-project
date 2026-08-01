//===- IdiomRecognizer.cpp - recognizing and raising idioms to CIR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for recognizing idioms (such as uses of functions
// and types to the C/C++ standard library) and replacing them with Clang IR
// operators for later optimization.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_IDIOMRECOGNIZER
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// True when the call's no builtin state forbids treating it as `name`. A
// builtin mark wins over a nobuiltin mark or a nobuiltins list.
bool isNoBuiltin(CallOp call, llvm::StringRef name) {
  if (call->hasAttr(cir::CIRDialect::getBuiltinAttrName()))
    return false;
  if (call->hasAttr(cir::CIRDialect::getNoBuiltinAttrName()))
    return true;
  auto noBuiltins = call->getAttrOfType<mlir::ArrayAttr>(
      cir::CIRDialect::getNoBuiltinsAttrName());
  if (!noBuiltins)
    return false;
  return noBuiltins.empty() ||
         llvm::any_of(noBuiltins, [name](mlir::Attribute entry) {
           auto builtinName = mlir::dyn_cast<mlir::StringAttr>(entry);
           return builtinName && builtinName.getValue() == name;
         });
}

// Raises a direct cir.call to the first candidate in `TargetOps` that matches.
template <typename... TargetOps> class StdRecognizer {
  template <typename TargetOp, size_t... Indices>
  static TargetOp buildCall(cir::CIRBaseBuilderTy &builder, CallOp call,
                            std::index_sequence<Indices...>) {
    return TargetOp::create(builder, call.getLoc(),
                            call->getResult(0).getType(),
                            call.getOperand(Indices)..., call.getCalleeAttr());
  }

  template <typename TargetOp>
  static bool raiseOne(CallOp call, mlir::MLIRContext &context,
                       mlir::SymbolTableCollection &symbolTables) {
    // A musttail call must stay a call, so it is never raised.
    if (!call.getCallee() || call.getMusttail() ||
        !TargetOp::signatureMatches(call->getOperandTypes(),
                                    call->getResultTypes()))
      return false;

    if constexpr (TargetOp::hasKnownFuncKind()) {
      // Only a free std function with the right name carries the tag, so
      // members, static members, and operators never match. The shape of the
      // call is checked here, so a variadic callee never matches.
      cir::FuncOp callee = call.resolveCalleeInTable(symbolTables);
      if (!callee || callee.getFunctionType().isVarArg())
        return false;
      auto funcIdentity = mlir::dyn_cast_if_present<cir::FuncIdentityAttr>(
          callee.getFuncInfoAttr());
      if (!funcIdentity || funcIdentity.getKind() != TargetOp::getFuncKind())
        return false;
    } else {
      // A C library function has no identity tag, so it is matched by callee
      // symbol, which works because C names are unmangled. The symbol alone is
      // not enough when builtins are disabled, so the recorded no builtin state
      // gates the match.
      if (*call.getCallee() != TargetOp::getFunctionName() ||
          isNoBuiltin(call, TargetOp::getFunctionName()))
        return false;
      // The library function is not variadic, so a variadic callee that only
      // shares the name is not that function. This lookup runs only after the
      // name matches.
      cir::FuncOp callee = call.resolveCalleeInTable(symbolTables);
      if (callee && callee.getFunctionType().isVarArg())
        return false;
    }

    cir::CIRBaseBuilderTy builder(context);
    builder.setInsertionPointAfter(call.getOperation());
    constexpr unsigned numArgs = TargetOp::getNumArgs();
    TargetOp op =
        buildCall<TargetOp>(builder, call, std::make_index_sequence<numArgs>());
    // The raised operation keeps every call attribute except the callee,
    // which it carries as original_fn, so lowering back loses nothing.
    for (mlir::NamedAttribute attr : call->getAttrs())
      if (attr.getName() != call.getCalleeAttrName())
        op->setAttr(attr.getName(), attr.getValue());
    call.replaceAllUsesWith(op);
    call.erase();
    return true;
  }

public:
  // Tries each candidate in order and stops at the first that raises.
  static bool raise(CallOp call, mlir::MLIRContext &context,
                    mlir::SymbolTableCollection &symbolTables) {
    return (raiseOne<TargetOps>(call, context, symbolTables) || ...);
  }
};

// The library calls the recognizer knows how to raise, tried in order.
using RecognizedStdOps = StdRecognizer<StdFindOp, StrLenOp>;

struct IdiomRecognizerPass
    : public impl::IdiomRecognizerBase<IdiomRecognizerPass> {
  IdiomRecognizerPass() = default;

  void runOnOperation() override;

  void recognizeStandardLibraryCall(CallOp call,
                                    mlir::SymbolTableCollection &symbolTables);
};
} // namespace

void IdiomRecognizerPass::recognizeStandardLibraryCall(
    CallOp call, mlir::SymbolTableCollection &symbolTables) {
  RecognizedStdOps::raise(call, getContext(), symbolTables);
}

void IdiomRecognizerPass::runOnOperation() {
  // The facts this pass reads live on the operations, so it needs no AST
  // and also works on parsed CIR assembly.
  mlir::SymbolTableCollection symbolTables;

  getOperation()->walk([&](CallOp callOp) {
    // Skip indirect calls.
    std::optional<llvm::StringRef> callee = callOp.getCallee();
    if (!callee)
      return;

    recognizeStandardLibraryCall(callOp, symbolTables);
  });
}

std::unique_ptr<Pass> mlir::createIdiomRecognizerPass() {
  return std::make_unique<IdiomRecognizerPass>();
}
