//===--- ParseOpenMP.cpp - OpenMP directives parsing ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements parsing of all OpenMP directives and clauses.
///
//===----------------------------------------------------------------------===//

#include "RAIIObjectsForParser.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/PointerIntPair.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// OpenMP declarative directives.
//===----------------------------------------------------------------------===//

static OpenMPDirectiveKind ParseOpenMPDirectiveKind(Parser &P) {
  // Array of foldings: F[i][0] F[i][1] ===> F[i][2].
  // E.g.: OMPD_for OMPD_simd ===> OMPD_for_simd
  // TODO: add other combined directives in topological order.
  const OpenMPDirectiveKind F[][3] = {
      {OMPD_unknown /*cancellation*/, OMPD_unknown /*point*/,
       OMPD_cancellation_point},
      {OMPD_target, OMPD_unknown /*data*/, OMPD_target_data},
      {OMPD_for, OMPD_simd, OMPD_for_simd},
      {OMPD_parallel, OMPD_for, OMPD_parallel_for},
      {OMPD_parallel_for, OMPD_simd, OMPD_parallel_for_simd},
      {OMPD_parallel, OMPD_sections, OMPD_parallel_sections}};
  auto Tok = P.getCurToken();
  auto DKind =
      Tok.isAnnotation()
          ? OMPD_unknown
          : getOpenMPDirectiveKind(P.getPreprocessor().getSpelling(Tok));

  bool TokenMatched = false;
  for (unsigned i = 0; i < llvm::array_lengthof(F); ++i) {
    if (!Tok.isAnnotation() && DKind == OMPD_unknown) {
      TokenMatched =
          (i == 0) &&
          !P.getPreprocessor().getSpelling(Tok).compare("cancellation");
    } else {
      TokenMatched = DKind == F[i][0] && DKind != OMPD_unknown;
    }

    if (TokenMatched) {
      Tok = P.getPreprocessor().LookAhead(0);
      auto TokenIsAnnotation = Tok.isAnnotation();
      auto SDKind =
          TokenIsAnnotation
              ? OMPD_unknown
              : getOpenMPDirectiveKind(P.getPreprocessor().getSpelling(Tok));

      if (!TokenIsAnnotation && SDKind == OMPD_unknown) {
        TokenMatched =
            ((i == 0) &&
             !P.getPreprocessor().getSpelling(Tok).compare("point")) ||
            ((i == 1) && !P.getPreprocessor().getSpelling(Tok).compare("data"));
      } else {
        TokenMatched = SDKind == F[i][1] && SDKind != OMPD_unknown;
      }

      if (TokenMatched) {
        P.ConsumeToken();
        DKind = F[i][2];
      }
    }
  }
  return DKind;
}

/// \brief Parsing of declarative OpenMP directives.
///
///       threadprivate-directive:
///         annot_pragma_openmp 'threadprivate' simple-variable-list
///
Parser::DeclGroupPtrTy Parser::ParseOpenMPDeclarativeDirective() {
  assert(Tok.is(tok::annot_pragma_openmp) && "Not an OpenMP directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  SourceLocation Loc = ConsumeToken();
  SmallVector<Expr *, 5> Identifiers;
  auto DKind = ParseOpenMPDirectiveKind(*this);

  switch (DKind) {
  case OMPD_threadprivate:
    ConsumeToken();
    if (!ParseOpenMPSimpleVarList(OMPD_threadprivate, Identifiers, true)) {
      // The last seen token is annot_pragma_openmp_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_openmp_end)) {
        Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
            << getOpenMPDirectiveName(OMPD_threadprivate);
        SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
      }
      // Skip the last annot_pragma_openmp_end.
      ConsumeToken();
      return Actions.ActOnOpenMPThreadprivateDirective(Loc, Identifiers);
    }
    break;
  case OMPD_unknown:
    Diag(Tok, diag::err_omp_unknown_directive);
    break;
  case OMPD_parallel:
  case OMPD_simd:
  case OMPD_task:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_taskgroup:
  case OMPD_flush:
  case OMPD_for:
  case OMPD_for_simd:
  case OMPD_sections:
  case OMPD_section:
  case OMPD_single:
  case OMPD_master:
  case OMPD_ordered:
  case OMPD_critical:
  case OMPD_parallel_for:
  case OMPD_parallel_for_simd:
  case OMPD_parallel_sections:
  case OMPD_atomic:
  case OMPD_target:
  case OMPD_teams:
  case OMPD_cancellation_point:
  case OMPD_cancel:
  case OMPD_target_data:
    Diag(Tok, diag::err_omp_unexpected_directive)
        << getOpenMPDirectiveName(DKind);
    break;
  }
  SkipUntil(tok::annot_pragma_openmp_end);
  return DeclGroupPtrTy();
}

/// \brief Parsing of declarative or executable OpenMP directives.
///
///       threadprivate-directive:
///         annot_pragma_openmp 'threadprivate' simple-variable-list
///         annot_pragma_openmp_end
///
///       executable-directive:
///         annot_pragma_openmp 'parallel' | 'simd' | 'for' | 'sections' |
///         'section' | 'single' | 'master' | 'critical' [ '(' <name> ')' ] |
///         'parallel for' | 'parallel sections' | 'task' | 'taskyield' |
///         'barrier' | 'taskwait' | 'flush' | 'ordered' | 'atomic' |
///         'for simd' | 'parallel for simd' | 'target' | 'target data' |
///         'taskgroup' | 'teams' {clause}
///         annot_pragma_openmp_end
///
StmtResult
Parser::ParseOpenMPDeclarativeOrExecutableDirective(bool StandAloneAllowed) {
  assert(Tok.is(tok::annot_pragma_openmp) && "Not an OpenMP directive!");
  ParenBraceBracketBalancer BalancerRAIIObj(*this);
  SmallVector<Expr *, 5> Identifiers;
  SmallVector<OMPClause *, 5> Clauses;
  SmallVector<llvm::PointerIntPair<OMPClause *, 1, bool>, OMPC_unknown + 1>
  FirstClauses(OMPC_unknown + 1);
  unsigned ScopeFlags =
      Scope::FnScope | Scope::DeclScope | Scope::OpenMPDirectiveScope;
  SourceLocation Loc = ConsumeToken(), EndLoc;
  auto DKind = ParseOpenMPDirectiveKind(*this);
  OpenMPDirectiveKind CancelRegion = OMPD_unknown;
  // Name of critical directive.
  DeclarationNameInfo DirName;
  StmtResult Directive = StmtError();
  bool HasAssociatedStatement = true;
  bool FlushHasClause = false;

  switch (DKind) {
  case OMPD_threadprivate:
    ConsumeToken();
    if (!ParseOpenMPSimpleVarList(OMPD_threadprivate, Identifiers, false)) {
      // The last seen token is annot_pragma_openmp_end - need to check for
      // extra tokens.
      if (Tok.isNot(tok::annot_pragma_openmp_end)) {
        Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
            << getOpenMPDirectiveName(OMPD_threadprivate);
        SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
      }
      DeclGroupPtrTy Res =
          Actions.ActOnOpenMPThreadprivateDirective(Loc, Identifiers);
      Directive = Actions.ActOnDeclStmt(Res, Loc, Tok.getLocation());
    }
    SkipUntil(tok::annot_pragma_openmp_end);
    break;
  case OMPD_flush:
    if (PP.LookAhead(0).is(tok::l_paren)) {
      FlushHasClause = true;
      // Push copy of the current token back to stream to properly parse
      // pseudo-clause OMPFlushClause.
      PP.EnterToken(Tok);
    }
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_cancellation_point:
  case OMPD_cancel:
    if (!StandAloneAllowed) {
      Diag(Tok, diag::err_omp_immediate_directive)
          << getOpenMPDirectiveName(DKind);
    }
    HasAssociatedStatement = false;
    // Fall through for further analysis.
  case OMPD_parallel:
  case OMPD_simd:
  case OMPD_for:
  case OMPD_for_simd:
  case OMPD_sections:
  case OMPD_single:
  case OMPD_section:
  case OMPD_master:
  case OMPD_critical:
  case OMPD_parallel_for:
  case OMPD_parallel_for_simd:
  case OMPD_parallel_sections:
  case OMPD_task:
  case OMPD_ordered:
  case OMPD_atomic:
  case OMPD_target:
  case OMPD_teams:
  case OMPD_taskgroup:
  case OMPD_target_data: {
    ConsumeToken();
    // Parse directive name of the 'critical' directive if any.
    if (DKind == OMPD_critical) {
      BalancedDelimiterTracker T(*this, tok::l_paren,
                                 tok::annot_pragma_openmp_end);
      if (!T.consumeOpen()) {
        if (Tok.isAnyIdentifier()) {
          DirName =
              DeclarationNameInfo(Tok.getIdentifierInfo(), Tok.getLocation());
          ConsumeAnyToken();
        } else {
          Diag(Tok, diag::err_omp_expected_identifier_for_critical);
        }
        T.consumeClose();
      }
    } else if (DKind == OMPD_cancellation_point || DKind == OMPD_cancel) {
      CancelRegion = ParseOpenMPDirectiveKind(*this);
      if (Tok.isNot(tok::annot_pragma_openmp_end))
        ConsumeToken();
    }

    if (isOpenMPLoopDirective(DKind))
      ScopeFlags |= Scope::OpenMPLoopDirectiveScope;
    if (isOpenMPSimdDirective(DKind))
      ScopeFlags |= Scope::OpenMPSimdDirectiveScope;
    ParseScope OMPDirectiveScope(this, ScopeFlags);
    Actions.StartOpenMPDSABlock(DKind, DirName, Actions.getCurScope(), Loc);

    while (Tok.isNot(tok::annot_pragma_openmp_end)) {
      OpenMPClauseKind CKind =
          Tok.isAnnotation()
              ? OMPC_unknown
              : FlushHasClause ? OMPC_flush
                               : getOpenMPClauseKind(PP.getSpelling(Tok));
      Actions.StartOpenMPClause(CKind);
      FlushHasClause = false;
      OMPClause *Clause =
          ParseOpenMPClause(DKind, CKind, !FirstClauses[CKind].getInt());
      FirstClauses[CKind].setInt(true);
      if (Clause) {
        FirstClauses[CKind].setPointer(Clause);
        Clauses.push_back(Clause);
      }

      // Skip ',' if any.
      if (Tok.is(tok::comma))
        ConsumeToken();
      Actions.EndOpenMPClause();
    }
    // End location of the directive.
    EndLoc = Tok.getLocation();
    // Consume final annot_pragma_openmp_end.
    ConsumeToken();

    StmtResult AssociatedStmt;
    if (HasAssociatedStatement) {
      // The body is a block scope like in Lambdas and Blocks.
      Sema::CompoundScopeRAII CompoundScope(Actions);
      Actions.ActOnOpenMPRegionStart(DKind, getCurScope());
      Actions.ActOnStartOfCompoundStmt();
      // Parse statement
      AssociatedStmt = ParseStatement();
      Actions.ActOnFinishOfCompoundStmt();
      AssociatedStmt = Actions.ActOnOpenMPRegionEnd(AssociatedStmt, Clauses);
    }
    Directive = Actions.ActOnOpenMPExecutableDirective(
        DKind, DirName, CancelRegion, Clauses, AssociatedStmt.get(), Loc,
        EndLoc);

    // Exit scope.
    Actions.EndOpenMPDSABlock(Directive.get());
    OMPDirectiveScope.Exit();
    break;
  }
  case OMPD_unknown:
    Diag(Tok, diag::err_omp_unknown_directive);
    SkipUntil(tok::annot_pragma_openmp_end);
    break;
  }
  return Directive;
}

/// \brief Parses list of simple variables for '#pragma omp threadprivate'
/// directive.
///
///   simple-variable-list:
///         '(' id-expression {, id-expression} ')'
///
bool Parser::ParseOpenMPSimpleVarList(OpenMPDirectiveKind Kind,
                                      SmallVectorImpl<Expr *> &VarList,
                                      bool AllowScopeSpecifier) {
  VarList.clear();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPDirectiveName(Kind)))
    return true;
  bool IsCorrect = true;
  bool NoIdentIsFound = true;

  // Read tokens while ')' or annot_pragma_openmp_end is not found.
  while (Tok.isNot(tok::r_paren) && Tok.isNot(tok::annot_pragma_openmp_end)) {
    CXXScopeSpec SS;
    SourceLocation TemplateKWLoc;
    UnqualifiedId Name;
    // Read var name.
    Token PrevTok = Tok;
    NoIdentIsFound = false;

    if (AllowScopeSpecifier && getLangOpts().CPlusPlus &&
        ParseOptionalCXXScopeSpecifier(SS, ParsedType(), false)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    } else if (ParseUnqualifiedId(SS, false, false, false, ParsedType(),
                                  TemplateKWLoc, Name)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    } else if (Tok.isNot(tok::comma) && Tok.isNot(tok::r_paren) &&
               Tok.isNot(tok::annot_pragma_openmp_end)) {
      IsCorrect = false;
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
      Diag(PrevTok.getLocation(), diag::err_expected)
          << tok::identifier
          << SourceRange(PrevTok.getLocation(), PrevTokLocation);
    } else {
      DeclarationNameInfo NameInfo = Actions.GetNameFromUnqualifiedId(Name);
      ExprResult Res =
          Actions.ActOnOpenMPIdExpression(getCurScope(), SS, NameInfo);
      if (Res.isUsable())
        VarList.push_back(Res.get());
    }
    // Consume ','.
    if (Tok.is(tok::comma)) {
      ConsumeToken();
    }
  }

  if (NoIdentIsFound) {
    Diag(Tok, diag::err_expected) << tok::identifier;
    IsCorrect = false;
  }

  // Parse ')'.
  IsCorrect = !T.consumeClose() && IsCorrect;

  return !IsCorrect && VarList.empty();
}

/// \brief Parsing of OpenMP clauses.
///
///    clause:
///       if-clause | final-clause | num_threads-clause | safelen-clause |
///       default-clause | private-clause | firstprivate-clause | shared-clause
///       | linear-clause | aligned-clause | collapse-clause |
///       lastprivate-clause | reduction-clause | proc_bind-clause |
///       schedule-clause | copyin-clause | copyprivate-clause | untied-clause |
///       mergeable-clause | flush-clause | read-clause | write-clause |
///       update-clause | capture-clause | seq_cst-clause | device-clause |
///       simdlen-clause | threads-clause | simd-clause
///
OMPClause *Parser::ParseOpenMPClause(OpenMPDirectiveKind DKind,
                                     OpenMPClauseKind CKind, bool FirstClause) {
  OMPClause *Clause = nullptr;
  bool ErrorFound = false;
  // Check if clause is allowed for the given directive.
  if (CKind != OMPC_unknown && !isAllowedClauseForDirective(DKind, CKind)) {
    Diag(Tok, diag::err_omp_unexpected_clause) << getOpenMPClauseName(CKind)
                                               << getOpenMPDirectiveName(DKind);
    ErrorFound = true;
  }

  switch (CKind) {
  case OMPC_final:
  case OMPC_num_threads:
  case OMPC_safelen:
  case OMPC_simdlen:
  case OMPC_collapse:
  case OMPC_ordered:
  case OMPC_device:
    // OpenMP [2.5, Restrictions]
    //  At most one num_threads clause can appear on the directive.
    // OpenMP [2.8.1, simd construct, Restrictions]
    //  Only one safelen  clause can appear on a simd directive.
    //  Only one simdlen  clause can appear on a simd directive.
    //  Only one collapse clause can appear on a simd directive.
    // OpenMP [2.9.1, target data construct, Restrictions]
    //  At most one device clause can appear on the directive.
    // OpenMP [2.11.1, task Construct, Restrictions]
    //  At most one if clause can appear on the directive.
    //  At most one final clause can appear on the directive.
    if (!FirstClause) {
      Diag(Tok, diag::err_omp_more_one_clause)
          << getOpenMPDirectiveName(DKind) << getOpenMPClauseName(CKind) << 0;
      ErrorFound = true;
    }

    if (CKind == OMPC_ordered && PP.LookAhead(/*N=*/0).isNot(tok::l_paren))
      Clause = ParseOpenMPClause(CKind);
    else
      Clause = ParseOpenMPSingleExprClause(CKind);
    break;
  case OMPC_default:
  case OMPC_proc_bind:
    // OpenMP [2.14.3.1, Restrictions]
    //  Only a single default clause may be specified on a parallel, task or
    //  teams directive.
    // OpenMP [2.5, parallel Construct, Restrictions]
    //  At most one proc_bind clause can appear on the directive.
    if (!FirstClause) {
      Diag(Tok, diag::err_omp_more_one_clause)
          << getOpenMPDirectiveName(DKind) << getOpenMPClauseName(CKind) << 0;
      ErrorFound = true;
    }

    Clause = ParseOpenMPSimpleClause(CKind);
    break;
  case OMPC_schedule:
    // OpenMP [2.7.1, Restrictions, p. 3]
    //  Only one schedule clause can appear on a loop directive.
    if (!FirstClause) {
      Diag(Tok, diag::err_omp_more_one_clause)
          << getOpenMPDirectiveName(DKind) << getOpenMPClauseName(CKind) << 0;
      ErrorFound = true;
    }

  case OMPC_if:
    Clause = ParseOpenMPSingleExprWithArgClause(CKind);
    break;
  case OMPC_nowait:
  case OMPC_untied:
  case OMPC_mergeable:
  case OMPC_read:
  case OMPC_write:
  case OMPC_update:
  case OMPC_capture:
  case OMPC_seq_cst:
  case OMPC_threads:
  case OMPC_simd:
    // OpenMP [2.7.1, Restrictions, p. 9]
    //  Only one ordered clause can appear on a loop directive.
    // OpenMP [2.7.1, Restrictions, C/C++, p. 4]
    //  Only one nowait clause can appear on a for directive.
    if (!FirstClause) {
      Diag(Tok, diag::err_omp_more_one_clause)
          << getOpenMPDirectiveName(DKind) << getOpenMPClauseName(CKind) << 0;
      ErrorFound = true;
    }

    Clause = ParseOpenMPClause(CKind);
    break;
  case OMPC_private:
  case OMPC_firstprivate:
  case OMPC_lastprivate:
  case OMPC_shared:
  case OMPC_reduction:
  case OMPC_linear:
  case OMPC_aligned:
  case OMPC_copyin:
  case OMPC_copyprivate:
  case OMPC_flush:
  case OMPC_depend:
  case OMPC_map:
    Clause = ParseOpenMPVarListClause(CKind);
    break;
  case OMPC_unknown:
    Diag(Tok, diag::warn_omp_extra_tokens_at_eol)
        << getOpenMPDirectiveName(DKind);
    SkipUntil(tok::annot_pragma_openmp_end, StopBeforeMatch);
    break;
  case OMPC_threadprivate:
    Diag(Tok, diag::err_omp_unexpected_clause) << getOpenMPClauseName(CKind)
                                               << getOpenMPDirectiveName(DKind);
    SkipUntil(tok::comma, tok::annot_pragma_openmp_end, StopBeforeMatch);
    break;
  }
  return ErrorFound ? nullptr : Clause;
}

/// \brief Parsing of OpenMP clauses with single expressions like 'final',
/// 'collapse', 'safelen', 'num_threads', 'simdlen', 'num_teams', 'thread_limit'
/// or 'simdlen'.
///
///    final-clause:
///      'final' '(' expression ')'
///
///    num_threads-clause:
///      'num_threads' '(' expression ')'
///
///    safelen-clause:
///      'safelen' '(' expression ')'
///
///    simdlen-clause:
///      'simdlen' '(' expression ')'
///
///    collapse-clause:
///      'collapse' '(' expression ')'
///
OMPClause *Parser::ParseOpenMPSingleExprClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  SourceLocation ELoc = Tok.getLocation();
  ExprResult LHS(ParseCastExpression(false, false, NotTypeCast));
  ExprResult Val(ParseRHSOfBinaryExpression(LHS, prec::Conditional));
  Val = Actions.ActOnFinishFullExpr(Val.get(), ELoc);

  // Parse ')'.
  T.consumeClose();

  if (Val.isInvalid())
    return nullptr;

  return Actions.ActOnOpenMPSingleExprClause(
      Kind, Val.get(), Loc, T.getOpenLocation(), T.getCloseLocation());
}

/// \brief Parsing of simple OpenMP clauses like 'default' or 'proc_bind'.
///
///    default-clause:
///         'default' '(' 'none' | 'shared' ')
///
///    proc_bind-clause:
///         'proc_bind' '(' 'master' | 'close' | 'spread' ')
///
OMPClause *Parser::ParseOpenMPSimpleClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  unsigned Type = getOpenMPSimpleClauseType(
      Kind, Tok.isAnnotation() ? "" : PP.getSpelling(Tok));
  SourceLocation TypeLoc = Tok.getLocation();
  if (Tok.isNot(tok::r_paren) && Tok.isNot(tok::comma) &&
      Tok.isNot(tok::annot_pragma_openmp_end))
    ConsumeAnyToken();

  // Parse ')'.
  T.consumeClose();

  return Actions.ActOnOpenMPSimpleClause(Kind, Type, TypeLoc, LOpen, Loc,
                                         Tok.getLocation());
}

/// \brief Parsing of OpenMP clauses like 'ordered'.
///
///    ordered-clause:
///         'ordered'
///
///    nowait-clause:
///         'nowait'
///
///    untied-clause:
///         'untied'
///
///    mergeable-clause:
///         'mergeable'
///
///    read-clause:
///         'read'
///
///    threads-clause:
///         'threads'
///
///    simd-clause:
///         'simd'
///
OMPClause *Parser::ParseOpenMPClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = Tok.getLocation();
  ConsumeAnyToken();

  return Actions.ActOnOpenMPClause(Kind, Loc, Tok.getLocation());
}


/// \brief Parsing of OpenMP clauses with single expressions and some additional
/// argument like 'schedule' or 'dist_schedule'.
///
///    schedule-clause:
///      'schedule' '(' kind [',' expression ] ')'
///
///    if-clause:
///      'if' '(' [ directive-name-modifier ':' ] expression ')'
///
OMPClause *Parser::ParseOpenMPSingleExprWithArgClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = ConsumeToken();
  SourceLocation DelimLoc;
  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  ExprResult Val;
  unsigned Arg;
  SourceLocation KLoc;
  if (Kind == OMPC_schedule) {
    Arg = getOpenMPSimpleClauseType(
        Kind, Tok.isAnnotation() ? "" : PP.getSpelling(Tok));
    KLoc = Tok.getLocation();
    if (Tok.isNot(tok::r_paren) && Tok.isNot(tok::comma) &&
        Tok.isNot(tok::annot_pragma_openmp_end))
      ConsumeAnyToken();
    if ((Arg == OMPC_SCHEDULE_static || Arg == OMPC_SCHEDULE_dynamic ||
         Arg == OMPC_SCHEDULE_guided) &&
        Tok.is(tok::comma))
      DelimLoc = ConsumeAnyToken();
  } else {
    assert(Kind == OMPC_if);
    KLoc = Tok.getLocation();
    Arg = ParseOpenMPDirectiveKind(*this);
    if (Arg != OMPD_unknown) {
      ConsumeToken();
      if (Tok.is(tok::colon))
        DelimLoc = ConsumeToken();
      else
        Diag(Tok, diag::warn_pragma_expected_colon)
            << "directive name modifier";
    }
  }

  bool NeedAnExpression =
      (Kind == OMPC_schedule && DelimLoc.isValid()) || Kind == OMPC_if;
  if (NeedAnExpression) {
    SourceLocation ELoc = Tok.getLocation();
    ExprResult LHS(ParseCastExpression(false, false, NotTypeCast));
    Val = ParseRHSOfBinaryExpression(LHS, prec::Conditional);
    Val = Actions.ActOnFinishFullExpr(Val.get(), ELoc);
  }

  // Parse ')'.
  T.consumeClose();

  if (NeedAnExpression && Val.isInvalid())
    return nullptr;

  return Actions.ActOnOpenMPSingleExprWithArgClause(
      Kind, Arg, Val.get(), Loc, T.getOpenLocation(), KLoc, DelimLoc,
      T.getCloseLocation());
}

static bool ParseReductionId(Parser &P, CXXScopeSpec &ReductionIdScopeSpec,
                             UnqualifiedId &ReductionId) {
  SourceLocation TemplateKWLoc;
  if (ReductionIdScopeSpec.isEmpty()) {
    auto OOK = OO_None;
    switch (P.getCurToken().getKind()) {
    case tok::plus:
      OOK = OO_Plus;
      break;
    case tok::minus:
      OOK = OO_Minus;
      break;
    case tok::star:
      OOK = OO_Star;
      break;
    case tok::amp:
      OOK = OO_Amp;
      break;
    case tok::pipe:
      OOK = OO_Pipe;
      break;
    case tok::caret:
      OOK = OO_Caret;
      break;
    case tok::ampamp:
      OOK = OO_AmpAmp;
      break;
    case tok::pipepipe:
      OOK = OO_PipePipe;
      break;
    default:
      break;
    }
    if (OOK != OO_None) {
      SourceLocation OpLoc = P.ConsumeToken();
      SourceLocation SymbolLocations[] = {OpLoc, OpLoc, SourceLocation()};
      ReductionId.setOperatorFunctionId(OpLoc, OOK, SymbolLocations);
      return false;
    }
  }
  return P.ParseUnqualifiedId(ReductionIdScopeSpec, /*EnteringContext*/ false,
                              /*AllowDestructorName*/ false,
                              /*AllowConstructorName*/ false, ParsedType(),
                              TemplateKWLoc, ReductionId);
}

/// \brief Parsing of OpenMP clause 'private', 'firstprivate', 'lastprivate',
/// 'shared', 'copyin', 'copyprivate', 'flush' or 'reduction'.
///
///    private-clause:
///       'private' '(' list ')'
///    firstprivate-clause:
///       'firstprivate' '(' list ')'
///    lastprivate-clause:
///       'lastprivate' '(' list ')'
///    shared-clause:
///       'shared' '(' list ')'
///    linear-clause:
///       'linear' '(' linear-list [ ':' linear-step ] ')'
///    aligned-clause:
///       'aligned' '(' list [ ':' alignment ] ')'
///    reduction-clause:
///       'reduction' '(' reduction-identifier ':' list ')'
///    copyprivate-clause:
///       'copyprivate' '(' list ')'
///    flush-clause:
///       'flush' '(' list ')'
///    depend-clause:
///       'depend' '(' in | out | inout : list ')'
///    map-clause:
///       'map' '(' [ [ always , ]
///          to | from | tofrom | alloc | release | delete ':' ] list ')';
///
/// For 'linear' clause linear-list may have the following forms:
///  list
///  modifier(list)
/// where modifier is 'val' (C) or 'ref', 'val' or 'uval'(C++).
OMPClause *Parser::ParseOpenMPVarListClause(OpenMPClauseKind Kind) {
  SourceLocation Loc = Tok.getLocation();
  SourceLocation LOpen = ConsumeToken();
  SourceLocation ColonLoc = SourceLocation();
  // Optional scope specifier and unqualified id for reduction identifier.
  CXXScopeSpec ReductionIdScopeSpec;
  UnqualifiedId ReductionId;
  bool InvalidReductionId = false;
  OpenMPDependClauseKind DepKind = OMPC_DEPEND_unknown;
  // OpenMP 4.1 [2.15.3.7, linear Clause]
  //  If no modifier is specified it is assumed to be val.
  OpenMPLinearClauseKind LinearModifier = OMPC_LINEAR_val;
  OpenMPMapClauseKind MapType = OMPC_MAP_unknown;
  OpenMPMapClauseKind MapTypeModifier = OMPC_MAP_unknown;
  bool MapTypeModifierSpecified = false;
  bool UnexpectedId = false;
  SourceLocation DepLinMapLoc;

  // Parse '('.
  BalancedDelimiterTracker T(*this, tok::l_paren, tok::annot_pragma_openmp_end);
  if (T.expectAndConsume(diag::err_expected_lparen_after,
                         getOpenMPClauseName(Kind)))
    return nullptr;

  bool NeedRParenForLinear = false;
  BalancedDelimiterTracker LinearT(*this, tok::l_paren,
                                  tok::annot_pragma_openmp_end);
  // Handle reduction-identifier for reduction clause.
  if (Kind == OMPC_reduction) {
    ColonProtectionRAIIObject ColonRAII(*this);
    if (getLangOpts().CPlusPlus) {
      ParseOptionalCXXScopeSpecifier(ReductionIdScopeSpec, ParsedType(), false);
    }
    InvalidReductionId =
        ParseReductionId(*this, ReductionIdScopeSpec, ReductionId);
    if (InvalidReductionId) {
      SkipUntil(tok::colon, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    }
    if (Tok.is(tok::colon)) {
      ColonLoc = ConsumeToken();
    } else {
      Diag(Tok, diag::warn_pragma_expected_colon) << "reduction identifier";
    }
  } else if (Kind == OMPC_depend) {
  // Handle dependency type for depend clause.
    ColonProtectionRAIIObject ColonRAII(*this);
    DepKind = static_cast<OpenMPDependClauseKind>(getOpenMPSimpleClauseType(
        Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
    DepLinMapLoc = Tok.getLocation();

    if (DepKind == OMPC_DEPEND_unknown) {
      SkipUntil(tok::colon, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    } else {
      ConsumeToken();
    }
    if (Tok.is(tok::colon)) {
      ColonLoc = ConsumeToken();
    } else {
      Diag(Tok, diag::warn_pragma_expected_colon) << "dependency type";
    }
  } else if (Kind == OMPC_linear) {
    // Try to parse modifier if any.
    if (Tok.is(tok::identifier) && PP.LookAhead(0).is(tok::l_paren)) {
      LinearModifier = static_cast<OpenMPLinearClauseKind>(
          getOpenMPSimpleClauseType(Kind, PP.getSpelling(Tok)));
      DepLinMapLoc = ConsumeToken();
      LinearT.consumeOpen();
      NeedRParenForLinear = true;
    }
  } else if (Kind == OMPC_map) {
    // Handle map type for map clause.
    ColonProtectionRAIIObject ColonRAII(*this);

    // the first identifier may be a list item, a map-type or
    //   a map-type-modifier
    MapType = static_cast<OpenMPMapClauseKind>(getOpenMPSimpleClauseType(
        Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
    DepLinMapLoc = Tok.getLocation();
    bool ColonExpected = false;

    if (Tok.is(tok::identifier)) {
      if (PP.LookAhead(0).is(tok::colon)) {
        MapType = static_cast<OpenMPMapClauseKind>(getOpenMPSimpleClauseType(
            Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
        if (MapType == OMPC_MAP_unknown) {
          Diag(Tok, diag::err_omp_unknown_map_type);
        } else if (MapType == OMPC_MAP_always) {
          Diag(Tok, diag::err_omp_map_type_missing);
        }
        ConsumeToken();
      } else if (PP.LookAhead(0).is(tok::comma)) {
        if (PP.LookAhead(1).is(tok::identifier) &&
            PP.LookAhead(2).is(tok::colon)) {
          MapTypeModifier =
              static_cast<OpenMPMapClauseKind>(getOpenMPSimpleClauseType(
                   Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
          if (MapTypeModifier != OMPC_MAP_always) {
            Diag(Tok, diag::err_omp_unknown_map_type_modifier);
            MapTypeModifier = OMPC_MAP_unknown;
          } else {
            MapTypeModifierSpecified = true;
          }

          ConsumeToken();
          ConsumeToken();

          MapType = static_cast<OpenMPMapClauseKind>(getOpenMPSimpleClauseType(
              Kind, Tok.is(tok::identifier) ? PP.getSpelling(Tok) : ""));
          if (MapType == OMPC_MAP_unknown || MapType == OMPC_MAP_always) {
            Diag(Tok, diag::err_omp_unknown_map_type);
          }
          ConsumeToken();
        } else {
          MapType = OMPC_MAP_tofrom;
        }
      } else {
        MapType = OMPC_MAP_tofrom;
      }
    } else {
      UnexpectedId = true;
    }

    if (Tok.is(tok::colon)) {
      ColonLoc = ConsumeToken();
    } else if (ColonExpected) {
      Diag(Tok, diag::warn_pragma_expected_colon) << "map type";
    }
  }

  SmallVector<Expr *, 5> Vars;
  bool IsComma =
      ((Kind != OMPC_reduction) && (Kind != OMPC_depend) &&
       (Kind != OMPC_map)) ||
      ((Kind == OMPC_reduction) && !InvalidReductionId) ||
      ((Kind == OMPC_map) && (UnexpectedId || MapType != OMPC_MAP_unknown) &&
       (!MapTypeModifierSpecified ||
        (MapTypeModifierSpecified && MapTypeModifier == OMPC_MAP_always))) ||
      ((Kind == OMPC_depend) && DepKind != OMPC_DEPEND_unknown);
  const bool MayHaveTail = (Kind == OMPC_linear || Kind == OMPC_aligned);
  while (IsComma || (Tok.isNot(tok::r_paren) && Tok.isNot(tok::colon) &&
                     Tok.isNot(tok::annot_pragma_openmp_end))) {
    ColonProtectionRAIIObject ColonRAII(*this, MayHaveTail);
    // Parse variable
    ExprResult VarExpr =
        Actions.CorrectDelayedTyposInExpr(ParseAssignmentExpression());
    if (VarExpr.isUsable()) {
      Vars.push_back(VarExpr.get());
    } else {
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
    }
    // Skip ',' if any
    IsComma = Tok.is(tok::comma);
    if (IsComma)
      ConsumeToken();
    else if (Tok.isNot(tok::r_paren) &&
             Tok.isNot(tok::annot_pragma_openmp_end) &&
             (!MayHaveTail || Tok.isNot(tok::colon)))
      Diag(Tok, diag::err_omp_expected_punc)
          << ((Kind == OMPC_flush) ? getOpenMPDirectiveName(OMPD_flush)
                                   : getOpenMPClauseName(Kind))
          << (Kind == OMPC_flush);
  }

  // Parse ')' for linear clause with modifier.
  if (NeedRParenForLinear)
    LinearT.consumeClose();

  // Parse ':' linear-step (or ':' alignment).
  Expr *TailExpr = nullptr;
  const bool MustHaveTail = MayHaveTail && Tok.is(tok::colon);
  if (MustHaveTail) {
    ColonLoc = Tok.getLocation();
    SourceLocation ELoc = ConsumeToken();
    ExprResult Tail = ParseAssignmentExpression();
    Tail = Actions.ActOnFinishFullExpr(Tail.get(), ELoc);
    if (Tail.isUsable())
      TailExpr = Tail.get();
    else
      SkipUntil(tok::comma, tok::r_paren, tok::annot_pragma_openmp_end,
                StopBeforeMatch);
  }

  // Parse ')'.
  T.consumeClose();
  if ((Kind == OMPC_depend && DepKind != OMPC_DEPEND_unknown && Vars.empty()) ||
      (Kind != OMPC_depend && Vars.empty()) || (MustHaveTail && !TailExpr) ||
      (Kind == OMPC_map && MapType == OMPC_MAP_unknown) ||
      InvalidReductionId) {
    return nullptr;
  }

  return Actions.ActOnOpenMPVarListClause(
      Kind, Vars, TailExpr, Loc, LOpen, ColonLoc, Tok.getLocation(),
      ReductionIdScopeSpec,
      ReductionId.isValid() ? Actions.GetNameFromUnqualifiedId(ReductionId)
                            : DeclarationNameInfo(),
      DepKind, LinearModifier, MapTypeModifier, MapType, DepLinMapLoc);
}

