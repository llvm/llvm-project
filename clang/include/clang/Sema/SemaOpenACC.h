//===----- SemaOpenACC.h - Semantic Analysis for OpenACC constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis for OpenACC constructs and
/// clauses.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAOPENACC_H
#define LLVM_CLANG_SEMA_SEMAOPENACC_H

#include "clang/AST/DeclGroup.h"
#include "clang/Basic/OpenACCKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include <variant>

namespace clang {
class OpenACCClause;

class SemaOpenACC : public SemaBase {
public:
  /// A type to represent all the data for an OpenACC Clause that has been
  /// parsed, but not yet created/semantically analyzed. This is effectively a
  /// discriminated union on the 'Clause Kind', with all of the individual
  /// clause details stored in a std::variant.
  class OpenACCParsedClause {
    OpenACCDirectiveKind DirKind;
    OpenACCClauseKind ClauseKind;
    SourceRange ClauseRange;
    SourceLocation LParenLoc;

    struct DefaultDetails {
      OpenACCDefaultClauseKind DefaultClauseKind;
    };

    struct ConditionDetails {
      Expr *ConditionExpr;
    };

    struct IntExprDetails {
      SmallVector<Expr *> IntExprs;
    };

    struct VarListDetails {
      SmallVector<Expr *> VarList;
    };

    std::variant<std::monostate, DefaultDetails, ConditionDetails,
                 IntExprDetails, VarListDetails>
        Details = std::monostate{};

  public:
    OpenACCParsedClause(OpenACCDirectiveKind DirKind,
                        OpenACCClauseKind ClauseKind, SourceLocation BeginLoc)
        : DirKind(DirKind), ClauseKind(ClauseKind), ClauseRange(BeginLoc, {}) {}

    OpenACCDirectiveKind getDirectiveKind() const { return DirKind; }

    OpenACCClauseKind getClauseKind() const { return ClauseKind; }

    SourceLocation getBeginLoc() const { return ClauseRange.getBegin(); }

    SourceLocation getLParenLoc() const { return LParenLoc; }

    SourceLocation getEndLoc() const { return ClauseRange.getEnd(); }

    OpenACCDefaultClauseKind getDefaultClauseKind() const {
      assert(ClauseKind == OpenACCClauseKind::Default &&
             "Parsed clause is not a default clause");
      return std::get<DefaultDetails>(Details).DefaultClauseKind;
    }

    const Expr *getConditionExpr() const {
      return const_cast<OpenACCParsedClause *>(this)->getConditionExpr();
    }

    Expr *getConditionExpr() {
      assert((ClauseKind == OpenACCClauseKind::If ||
              (ClauseKind == OpenACCClauseKind::Self &&
               DirKind != OpenACCDirectiveKind::Update)) &&
             "Parsed clause kind does not have a condition expr");

      // 'self' has an optional ConditionExpr, so be tolerant of that. This will
      // assert in variant otherwise.
      if (ClauseKind == OpenACCClauseKind::Self &&
          std::holds_alternative<std::monostate>(Details))
        return nullptr;

      return std::get<ConditionDetails>(Details).ConditionExpr;
    }

    unsigned getNumIntExprs() const {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");
      return std::get<IntExprDetails>(Details).IntExprs.size();
    }

    ArrayRef<Expr *> getIntExprs() {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");
      return std::get<IntExprDetails>(Details).IntExprs;
    }

    ArrayRef<Expr *> getIntExprs() const {
      return const_cast<OpenACCParsedClause *>(this)->getIntExprs();
    }

    ArrayRef<Expr *> getVarList() {
      assert(ClauseKind == OpenACCClauseKind::Private &&
             "Parsed clause kind does not have a var-list");
      return std::get<VarListDetails>(Details).VarList;
    }

    ArrayRef<Expr *> getVarList() const {
      return const_cast<OpenACCParsedClause *>(this)->getVarList();
    }

    void setLParenLoc(SourceLocation EndLoc) { LParenLoc = EndLoc; }
    void setEndLoc(SourceLocation EndLoc) { ClauseRange.setEnd(EndLoc); }

    void setDefaultDetails(OpenACCDefaultClauseKind DefKind) {
      assert(ClauseKind == OpenACCClauseKind::Default &&
             "Parsed clause is not a default clause");
      Details = DefaultDetails{DefKind};
    }

    void setConditionDetails(Expr *ConditionExpr) {
      assert((ClauseKind == OpenACCClauseKind::If ||
              (ClauseKind == OpenACCClauseKind::Self &&
               DirKind != OpenACCDirectiveKind::Update)) &&
             "Parsed clause kind does not have a condition expr");
      // In C++ we can count on this being a 'bool', but in C this gets left as
      // some sort of scalar that codegen will have to take care of converting.
      assert((!ConditionExpr || ConditionExpr->isInstantiationDependent() ||
              ConditionExpr->getType()->isScalarType()) &&
             "Condition expression type not scalar/dependent");

      Details = ConditionDetails{ConditionExpr};
    }

    void setIntExprDetails(ArrayRef<Expr *> IntExprs) {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");
      Details = IntExprDetails{{IntExprs.begin(), IntExprs.end()}};
    }
    void setIntExprDetails(llvm::SmallVector<Expr *> &&IntExprs) {
      assert((ClauseKind == OpenACCClauseKind::NumGangs ||
              ClauseKind == OpenACCClauseKind::NumWorkers ||
              ClauseKind == OpenACCClauseKind::VectorLength) &&
             "Parsed clause kind does not have a int exprs");
      Details = IntExprDetails{std::move(IntExprs)};
    }

    void setVarListDetails(ArrayRef<Expr *> VarList) {
      assert(ClauseKind == OpenACCClauseKind::Private &&
             "Parsed clause kind does not have a var-list");
      Details = VarListDetails{{VarList.begin(), VarList.end()}};
    }

    void setVarListDetails(llvm::SmallVector<Expr *> &&VarList) {
      assert(ClauseKind == OpenACCClauseKind::Private &&
             "Parsed clause kind does not have a var-list");
      Details = VarListDetails{std::move(VarList)};
    }
  };

  SemaOpenACC(Sema &S);

  /// Called after parsing an OpenACC Clause so that it can be checked.
  OpenACCClause *ActOnClause(ArrayRef<const OpenACCClause *> ExistingClauses,
                             OpenACCParsedClause &Clause);

  /// Called after the construct has been parsed, but clauses haven't been
  /// parsed.  This allows us to diagnose not-implemented, as well as set up any
  /// state required for parsing the clauses.
  void ActOnConstruct(OpenACCDirectiveKind K, SourceLocation StartLoc);

  /// Called after the directive, including its clauses, have been parsed and
  /// parsing has consumed the 'annot_pragma_openacc_end' token. This DOES
  /// happen before any associated declarations or statements have been parsed.
  /// This function is only called when we are parsing a 'statement' context.
  bool ActOnStartStmtDirective(OpenACCDirectiveKind K, SourceLocation StartLoc);

  /// Called after the directive, including its clauses, have been parsed and
  /// parsing has consumed the 'annot_pragma_openacc_end' token. This DOES
  /// happen before any associated declarations or statements have been parsed.
  /// This function is only called when we are parsing a 'Decl' context.
  bool ActOnStartDeclDirective(OpenACCDirectiveKind K, SourceLocation StartLoc);
  /// Called when we encounter an associated statement for our construct, this
  /// should check legality of the statement as it appertains to this Construct.
  StmtResult ActOnAssociatedStmt(OpenACCDirectiveKind K, StmtResult AssocStmt);

  /// Called after the directive has been completely parsed, including the
  /// declaration group or associated statement.
  StmtResult ActOnEndStmtDirective(OpenACCDirectiveKind K,
                                   SourceLocation StartLoc,
                                   SourceLocation EndLoc,
                                   ArrayRef<OpenACCClause *> Clauses,
                                   StmtResult AssocStmt);

  /// Called after the directive has been completely parsed, including the
  /// declaration group or associated statement.
  DeclGroupRef ActOnEndDeclDirective();

  /// Called when encountering an 'int-expr' for OpenACC, and manages
  /// conversions and diagnostics to 'int'.
  ExprResult ActOnIntExpr(OpenACCDirectiveKind DK, OpenACCClauseKind CK,
                          SourceLocation Loc, Expr *IntExpr);

  /// Called when encountering a 'var' for OpenACC, ensures it is actually a
  /// declaration reference to a variable of the correct type.
  ExprResult ActOnVar(Expr *VarExpr);

  /// Checks and creates an Array Section used in an OpenACC construct/clause.
  ExprResult ActOnArraySectionExpr(Expr *Base, SourceLocation LBLoc,
                                   Expr *LowerBound,
                                   SourceLocation ColonLocFirst, Expr *Length,
                                   SourceLocation RBLoc);
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAOPENACC_H
