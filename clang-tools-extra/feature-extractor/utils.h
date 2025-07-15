#pragma once

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include <optional>

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
