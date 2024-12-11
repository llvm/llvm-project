#include "clang/Sema/DynamicCountPointerAssignmentAnalysisExported.h"
#include "DynamicCountPointerAssignmentAnalysis.h"
#include "TreeTransform.h"
#include "clang/AST/Expr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

ExprResult ReplaceCountExprParamsWithArgsFromCall(const Expr *CountExpr,
                                                  const CallExpr *CE, Sema &S) {
  // FIXME: Use of `const_cast` is here because it is hard to make
  // TreeTransform work with `const Expr` but we also want to provide a sane
  // public interface.
  CallExpr *CENoConst = const_cast<CallExpr *>(CE);
  SmallVector<Expr *, 4> CallArgs(
      CENoConst->getArgs(), CENoConst->getArgs() + CENoConst->getNumArgs());
  TransformDynamicCountWithFunctionArgument T(S, CallArgs);
  ExprResult Replaced = T.TransformExpr(const_cast<Expr *>(CountExpr));
  return Replaced;
}
} // namespace clang
