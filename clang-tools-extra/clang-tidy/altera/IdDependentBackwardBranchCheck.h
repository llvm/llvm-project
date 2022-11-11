//===--- IdDependentBackwardBranchCheck.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_IDDEPENDENTBACKWARDBRANCHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_IDDEPENDENTBACKWARDBRANCHCHECK_H

#include "../ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace altera {

/// Finds ID-dependent variables and fields used within loops, and warns of
/// their usage. Using these variables in loops can lead to performance
/// degradation.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/altera/id-dependent-backward-branch.html
class IdDependentBackwardBranchCheck : public ClangTidyCheck {
private:
  enum LoopType { UnknownLoop = -1, DoLoop = 0, WhileLoop = 1, ForLoop = 2 };
  // Stores information necessary for printing out source of error.
  struct IdDependencyRecord {
    IdDependencyRecord(const VarDecl *Declaration, SourceLocation Location,
                       const llvm::Twine &Message)
        : VariableDeclaration(Declaration), Location(Location),
          Message(Message.str()) {}
    IdDependencyRecord(const FieldDecl *Declaration, SourceLocation Location,
                       const llvm::Twine &Message)
        : FieldDeclaration(Declaration), Location(Location),
          Message(Message.str()) {}
    IdDependencyRecord() = default;
    const VarDecl *VariableDeclaration = nullptr;
    const FieldDecl *FieldDeclaration = nullptr;
    SourceLocation Location;
    std::string Message;
  };
  // Stores the locations where ID-dependent variables are created.
  std::map<const VarDecl *, IdDependencyRecord> IdDepVarsMap;
  // Stores the locations where ID-dependent fields are created.
  std::map<const FieldDecl *, IdDependencyRecord> IdDepFieldsMap;
  /// Returns an IdDependencyRecord if the Expression contains an ID-dependent
  /// variable, returns a nullptr otherwise.
  IdDependencyRecord *hasIdDepVar(const Expr *Expression);
  /// Returns an IdDependencyRecord if the Expression contains an ID-dependent
  /// field, returns a nullptr otherwise.
  IdDependencyRecord *hasIdDepField(const Expr *Expression);
  /// Stores the location an ID-dependent variable is created from a call to
  /// an ID function in IdDepVarsMap.
  void saveIdDepVar(const Stmt *Statement, const VarDecl *Variable);
  /// Stores the location an ID-dependent field is created from a call to an ID
  /// function in IdDepFieldsMap.
  void saveIdDepField(const Stmt *Statement, const FieldDecl *Field);
  /// Stores the location an ID-dependent variable is created from a reference
  /// to another ID-dependent variable or field in IdDepVarsMap.
  void saveIdDepVarFromReference(const DeclRefExpr *RefExpr,
                                 const MemberExpr *MemExpr,
                                 const VarDecl *PotentialVar);
  /// Stores the location an ID-dependent field is created from a reference to
  /// another ID-dependent variable or field in IdDepFieldsMap.
  void saveIdDepFieldFromReference(const DeclRefExpr *RefExpr,
                                   const MemberExpr *MemExpr,
                                   const FieldDecl *PotentialField);
  /// Returns the loop type.
  LoopType getLoopType(const Stmt *Loop);

public:
  IdDependentBackwardBranchCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace altera
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_ALTERA_IDDEPENDENTBACKWARDBRANCHCHECK_H
