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
#include "CIRGenConstantEmitter.h"
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

template <typename Op>
static RValue emitBuiltinBitOp(CIRGenFunction &cgf, const CallExpr *e,
                               bool poisonZero = false) {
  assert(!cir::MissingFeatures::builtinCheckKind());

  mlir::Value arg = cgf.emitScalarExpr(e->getArg(0));
  CIRGenBuilderTy &builder = cgf.getBuilder();

  Op op;
  if constexpr (std::is_same_v<Op, cir::BitClzOp> ||
                std::is_same_v<Op, cir::BitCtzOp>)
    op = builder.create<Op>(cgf.getLoc(e->getSourceRange()), arg, poisonZero);
  else
    op = builder.create<Op>(cgf.getLoc(e->getSourceRange()), arg);

  mlir::Value result = op.getResult();
  mlir::Type exprTy = cgf.convertType(e->getType());
  if (exprTy != result.getType())
    result = builder.createIntCast(result, exprTy);

  return RValue::get(result);
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

  assert(!cir::MissingFeatures::builtinCallF128());

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  unsigned builtinIDIfNoAsmLabel = fd->hasAttr<AsmLabelAttr>() ? 0 : builtinID;

  assert(!cir::MissingFeatures::builtinCallMathErrno());
  assert(!cir::MissingFeatures::builtinCall());

  mlir::Location loc = getLoc(e->getExprLoc());

  switch (builtinIDIfNoAsmLabel) {
  default:
    break;

  case Builtin::BI__assume:
  case Builtin::BI__builtin_assume: {
    if (e->getArg(0)->HasSideEffects(getContext()))
      return RValue::get(nullptr);

    mlir::Value argValue = emitCheckedArgForAssume(e->getArg(0));
    builder.create<cir::AssumeOp>(loc, argValue);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_complex: {
    mlir::Value real = emitScalarExpr(e->getArg(0));
    mlir::Value imag = emitScalarExpr(e->getArg(1));
    mlir::Value complex = builder.createComplexCreate(loc, real, imag);
    return RValue::get(complex);
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return emitBuiltinBitOp<cir::BitClrsbOp>(*this, e);

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzg:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitCtzOp>(*this, e, /*poisonZero=*/true);

  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzg:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitClzOp>(*this, e, /*poisonZero=*/true);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return emitBuiltinBitOp<cir::BitParityOp>(*this, e);

  case Builtin::BI__lzcnt16:
  case Builtin::BI__lzcnt:
  case Builtin::BI__lzcnt64:
    assert(!cir::MissingFeatures::builtinCheckKind());
    return emitBuiltinBitOp<cir::BitClzOp>(*this, e, /*poisonZero=*/false);

  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg:
    return emitBuiltinBitOp<cir::BitPopcountOp>(*this, e);

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability: {
    mlir::Value argValue = emitScalarExpr(e->getArg(0));
    mlir::Value expectedValue = emitScalarExpr(e->getArg(1));

    mlir::FloatAttr probAttr;
    if (builtinIDIfNoAsmLabel == Builtin::BI__builtin_expect_with_probability) {
      llvm::APFloat probability(0.0);
      const Expr *probArg = e->getArg(2);
      [[maybe_unused]] bool evalSucceeded =
          probArg->EvaluateAsFloat(probability, cgm.getASTContext());
      assert(evalSucceeded &&
             "probability should be able to evaluate as float");
      bool loseInfo = false; // ignored
      probability.convert(llvm::APFloat::IEEEdouble(),
                          llvm::RoundingMode::Dynamic, &loseInfo);
      probAttr = mlir::FloatAttr::get(mlir::Float64Type::get(&getMLIRContext()),
                                      probability);
    }

    auto result = builder.create<cir::ExpectOp>(getLoc(e->getSourceRange()),
                                                argValue.getType(), argValue,
                                                expectedValue, probAttr);
    return RValue::get(result);
  }
  }

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

mlir::Value CIRGenFunction::emitCheckedArgForAssume(const Expr *e) {
  mlir::Value argValue = evaluateExprAsBool(e);
  if (!sanOpts.has(SanitizerKind::Builtin))
    return argValue;

  assert(!cir::MissingFeatures::sanitizers());
  cgm.errorNYI(e->getSourceRange(),
               "emitCheckedArgForAssume: sanitizers are NYI");
  return {};
}
