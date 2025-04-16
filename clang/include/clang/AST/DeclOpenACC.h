//=- DeclOpenACC.h - Classes for representing OpenACC directives -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines OpenACC nodes for declarative directives.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLOPENACC_H
#define LLVM_CLANG_AST_DECLOPENACC_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/OpenACCClause.h"
#include "clang/Basic/OpenACCKinds.h"

namespace clang {

// A base class for the declaration constructs, which manages the clauses and
// basic source location information. Currently not part of the Decl inheritence
// tree, as we should never have a reason to store one of these.
class OpenACCConstructDecl : public Decl {
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
  // The directive kind, each implementation of this interface is expected to
  // handle a specific kind.
  OpenACCDirectiveKind DirKind = OpenACCDirectiveKind::Invalid;
  SourceLocation DirectiveLoc;
  SourceLocation EndLoc;
  /// The list of clauses.  This is stored here as an ArrayRef, as this is the
  /// most convienient place to access the list, however the list itself should
  /// be stored in leaf nodes, likely in trailing-storage.
  MutableArrayRef<const OpenACCClause *> Clauses;

protected:
  OpenACCConstructDecl(Kind DeclKind, DeclContext *DC, OpenACCDirectiveKind K,
                       SourceLocation StartLoc, SourceLocation DirLoc,
                       SourceLocation EndLoc)
      : Decl(DeclKind, DC, StartLoc), DirKind(K), DirectiveLoc(DirLoc),
        EndLoc(EndLoc) {}

  OpenACCConstructDecl(Kind DeclKind) : Decl(DeclKind, EmptyShell{}) {}

  void setClauseList(MutableArrayRef<const OpenACCClause *> NewClauses) {
    assert(Clauses.empty() && "Cannot change clause list");
    Clauses = NewClauses;
  }

public:
  OpenACCDirectiveKind getDirectiveKind() const { return DirKind; }
  SourceLocation getDirectiveLoc() const { return DirectiveLoc; }
  virtual SourceRange getSourceRange() const override LLVM_READONLY {
    return SourceRange(getLocation(), EndLoc);
  }

  ArrayRef<const OpenACCClause *> clauses() const { return Clauses; }
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K);
};

class OpenACCDeclareDecl final
    : public OpenACCConstructDecl,
      private llvm::TrailingObjects<OpenACCDeclareDecl, const OpenACCClause *> {
  friend TrailingObjects;
  friend class ASTDeclReader;
  friend class ASTDeclWriter;

  OpenACCDeclareDecl(unsigned NumClauses)
      : OpenACCConstructDecl(OpenACCDeclare) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCDeclareDecl(DeclContext *DC, SourceLocation StartLoc,
                     SourceLocation DirLoc, SourceLocation EndLoc,
                     ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructDecl(OpenACCDeclare, DC, OpenACCDirectiveKind::Declare,
                             StartLoc, DirLoc, EndLoc) {
    // Initialize the trailing storage.
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());

    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static OpenACCDeclareDecl *Create(ASTContext &Ctx, DeclContext *DC,
                                    SourceLocation StartLoc,
                                    SourceLocation DirLoc,
                                    SourceLocation EndLoc,
                                    ArrayRef<const OpenACCClause *> Clauses);
  static OpenACCDeclareDecl *
  CreateDeserialized(ASTContext &Ctx, GlobalDeclID ID, unsigned NumClauses);
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OpenACCDeclare; }
};

// Reprents a 'routine' directive with a name. When this has no name, it is
// represented as an attribute.
class OpenACCRoutineDecl final
    : public OpenACCConstructDecl,
      private llvm::TrailingObjects<OpenACCRoutineDecl, const OpenACCClause *> {
  friend TrailingObjects;
  friend class ASTDeclReader;
  friend class ASTDeclWriter;

  Expr *FuncRef = nullptr;
  SourceRange ParensLoc;

  OpenACCRoutineDecl(unsigned NumClauses)
      : OpenACCConstructDecl(OpenACCRoutine) {
    std::uninitialized_value_construct(
        getTrailingObjects<const OpenACCClause *>(),
        getTrailingObjects<const OpenACCClause *>() + NumClauses);
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  NumClauses));
  }

  OpenACCRoutineDecl(DeclContext *DC, SourceLocation StartLoc,
                     SourceLocation DirLoc, SourceLocation LParenLoc,
                     Expr *FuncRef, SourceLocation RParenLoc,
                     SourceLocation EndLoc,
                     ArrayRef<const OpenACCClause *> Clauses)
      : OpenACCConstructDecl(OpenACCRoutine, DC, OpenACCDirectiveKind::Routine,
                             StartLoc, DirLoc, EndLoc),
        FuncRef(FuncRef), ParensLoc(LParenLoc, RParenLoc) {
    assert(LParenLoc.isValid() &&
           "Cannot represent implicit name with this declaration");
    // Initialize the trailing storage.
    std::uninitialized_copy(Clauses.begin(), Clauses.end(),
                            getTrailingObjects<const OpenACCClause *>());
    setClauseList(MutableArrayRef(getTrailingObjects<const OpenACCClause *>(),
                                  Clauses.size()));
  }

public:
  static OpenACCRoutineDecl *
  Create(ASTContext &Ctx, DeclContext *DC, SourceLocation StartLoc,
         SourceLocation DirLoc, SourceLocation LParenLoc, Expr *FuncRef,
         SourceLocation RParenLoc, SourceLocation EndLoc,
         ArrayRef<const OpenACCClause *> Clauses);
  static OpenACCRoutineDecl *
  CreateDeserialized(ASTContext &Ctx, GlobalDeclID ID, unsigned NumClauses);
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == OpenACCRoutine; }

  const Expr *getFunctionReference() const { return FuncRef; }
  Expr *getFunctionReference() { return FuncRef; }

  SourceLocation getLParenLoc() const { return ParensLoc.getBegin(); }
  SourceLocation getRParenLoc() const { return ParensLoc.getEnd(); }
};
} // namespace clang

#endif
