#pragma once

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include <optional>

namespace Utils {

using namespace clang;

///
/// Check if two ValueDecl pointers refer to the same variable in AST
///
static inline bool are_same_variable(const ValueDecl *First,
                                     const ValueDecl *Second) {
  return First && Second &&
         First->getCanonicalDecl() == Second->getCanonicalDecl();
}

///
/// Check if for statement is defined in the translation unit file (not headers)
///
static inline bool is_in_main_file(ASTContext *context, const ForStmt *fs) {
  return fs && context->getSourceManager().isWrittenInMainFile(fs->getForLoc());
}

///
/// For a given for statement, tries to extract loop bound in the condition
/// This function extracts literal conditions from for loop condition
///
static inline std::optional<llvm::APInt>
get_for_condition_range_value(const ForStmt *fs) {
  const Expr *cond = fs->getCond();

  if (cond) {
    if (const BinaryOperator *BO = dyn_cast<BinaryOperator>(cond)) {
      const Expr *RHS = BO->getRHS()->IgnoreParenImpCasts();

      if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(RHS))
        return IL->getValue();
    }
  }

  return std::nullopt;
}

///
/// For a given for statement, tries to extract loop bound in the condition
/// This function evaluates macro conditions from for loop condition
///
static inline std::optional<llvm::APSInt>
get_for_condition_range_value(ASTContext *context, const ForStmt *fs) {

  if (const Expr *cond = fs->getCond(); cond) {
    if (const BinaryOperator *binOp = dyn_cast<BinaryOperator>(cond)) {
      const Expr *rhs = binOp->getRHS();
      clang::Expr::EvalResult eval;
      if (rhs->EvaluateAsInt(eval, *context)) {
        return eval.Val.getInt();
      }
    }
  }

  return std::nullopt;
}

///
/// For a given for statement, tries to extract loop bound in the condition. Use
/// this function instead of two previous ones. This one internally uses the
/// others
///
static inline std::optional<llvm::APInt>
maybe_get_for_bound(ASTContext *context, const ForStmt *fs) {
  if (const auto method1 = get_for_condition_range_value(fs))
    return llvm::APInt(64, method1.value().getSExtValue());
  else if (const auto method2 = get_for_condition_range_value(context, fs))
    return method2.value();
  else
    return std::nullopt;
}

///
/// For a given Stmt \s, tries to return the nearest ancestor of type
/// StatementType. Return nullptr in case no parent of given type was found.
///
template <typename StatementType>
static inline const StatementType *get_parent_stmt(ASTContext *context,
                                                   const Stmt *s) {
  auto parents = context->getParents(*s);

  if (parents.empty())
    return nullptr;

  const auto p = parents[0];

  if (const StatementType *parent_stmt = p.get<StatementType>())
    return parent_stmt;
  else if (const auto pStmt = p.get<Stmt>())
    return get_parent_stmt<StatementType>(context, pStmt);

  return nullptr;
}

///
/// Run a callable on all parents of type StatementType of \s recursively goes
/// up.
///
template <typename StatementType, typename Func>
static inline void run_on_all_parents_of_type(ASTContext *context,
                                              const Stmt *s, Func &&f) {
  auto parent = get_parent_stmt<StatementType>(context, s);

  while (parent) {
    f(context, parent);
    parent = get_parent_stmt<StatementType>(context,
                                            dyn_cast<StatementType>(parent));
  }
}

///
/// Get repetition of each for loop, considering parent for loops. For example,
/// for the following two nested for loops, result for the first for is 10, and
/// the result of nested one is 200
///
/// for(int i = 0; i < 10; i++)
///   for(int j = 0; j < 20; j++)
///   {}
///
llvm::APInt get_total_for_repetition_count(ASTContext *context,
                                           const ForStmt *fs) {
  auto bounds = maybe_get_for_bound(context, fs).value_or(llvm::APInt(64, 1));

  run_on_all_parents_of_type<ForStmt>(
      context, fs, [&bounds](auto ctx, auto fss) {
        bounds *= maybe_get_for_bound(ctx, fss).value_or(llvm::APInt(64, 1));
      });

  return bounds;
}

} // namespace Utils
