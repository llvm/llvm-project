//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as CIR or a function call to be
// later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenValue.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/Expr.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace llvm;

static RValue emitLibraryCall(CIRGenFunction &cgf, const FunctionDecl *fd,
                              const CallExpr *e, mlir::Operation *calleeValue) {
  CIRGenCallee callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(fd));
  return cgf.emitCall(e->getCallee()->getType(), callee, e, ReturnValueSlot());
}

RValue CIRGenFunction::emitBuiltinExpr(const GlobalDecl &gd, unsigned builtinID,
                                       const CallExpr *e,
                                       ReturnValueSlot returnValue) {
  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult result;
  if (e->isPRValue() && e->EvaluateAsRValue(result, cgm.getASTContext()) &&
      !result.hasSideEffects()) {
    if (result.Val.isInt()) {
      return RValue::get(builder.getConstInt(getLoc(e->getSourceRange()),
                                             result.Val.getInt()));
    }
    if (result.Val.isFloat()) {
      // Note: we are using result type of CallExpr to determine the type of
      // the constant. Classic codegen uses the result value to determine the
      // type. We feel it should be Ok to use expression type because it is
      // hard to imagine a builtin function evaluates to a value that
      // over/underflows its own defined type.
      mlir::Type type = convertType(e->getType());
      return RValue::get(builder.getConstFP(getLoc(e->getExprLoc()), type,
                                            result.Val.getFloat()));
    }
  }

  const FunctionDecl *fd = gd.getDecl()->getAsFunction();

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(builtinID))
    return emitLibraryCall(*this, fd, e,
                           cgm.getBuiltinLibFunction(fd, builtinID));

  cgm.errorNYI(e->getSourceRange(), "unimplemented builtin call");
  return getUndefRValue(e->getType());
}

/// Given a builtin id for a function like "__builtin_fabsf", return a Function*
/// for "fabsf".
cir::FuncOp CIRGenModule::getBuiltinLibFunction(const FunctionDecl *fd,
                                                unsigned builtinID) {
  assert(astContext.BuiltinInfo.isLibFunction(builtinID));

  // Get the name, skip over the __builtin_ prefix (if necessary). We may have
  // to build this up so provide a small stack buffer to handle the vast
  // majority of names.
  llvm::SmallString<64> name;

  assert(!cir::MissingFeatures::asmLabelAttr());
  name = astContext.BuiltinInfo.getName(builtinID).substr(10);

  GlobalDecl d(fd);
  mlir::Type type = convertType(fd->getType());
  return getOrCreateCIRFunction(name, type, d, /*forVTable=*/false);
}
