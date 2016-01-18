//===--- ParseStmt.cpp - Statement and Block Parser -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Statement and Block portions of the Parser
// interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "RAIIObjectsForParser.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/LoopHint.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/TypoCorrection.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.8: Statements and Blocks.
//===----------------------------------------------------------------------===//

/// \brief Parse a standalone statement (for instance, as the body of an 'if',
/// 'while', or 'for').
StmtResult Parser::ParseStatement(SourceLocation *TrailingElseLoc) {
  StmtResult Res;

  // We may get back a null statement if we found a #pragma. Keep going until
  // we get an actual statement.
  do {
    StmtVector Stmts;
    Res = ParseStatementOrDeclaration(Stmts, true, TrailingElseLoc);
  } while (!Res.isInvalid() && !Res.get());

  return Res;
}

/// ParseStatementOrDeclaration - Read 'statement' or 'declaration'.
///       StatementOrDeclaration:
///         statement
///         declaration
///
///       statement:
///         labeled-statement
///         compound-statement
///         expression-statement
///         selection-statement
///         iteration-statement
///         jump-statement
/// [C++]   declaration-statement
/// [C++]   try-block
/// [MS]    seh-try-block
/// [OBC]   objc-throw-statement
/// [OBC]   objc-try-catch-statement
/// [OBC]   objc-synchronized-statement
/// [GNU]   asm-statement
/// [OMP]   openmp-construct             [TODO]
///
///       labeled-statement:
///         identifier ':' statement
///         'case' constant-expression ':' statement
///         'default' ':' statement
///
///       selection-statement:
///         if-statement
///         switch-statement
///
///       iteration-statement:
///         while-statement
///         do-statement
///         for-statement
///
///       expression-statement:
///         expression[opt] ';'
///
///       jump-statement:
///         'goto' identifier ';'
///         'continue' ';'
///         'break' ';'
///         'return' expression[opt] ';'
/// [GNU]   'goto' '*' expression ';'
///
/// [OBC] objc-throw-statement:
/// [OBC]   '@' 'throw' expression ';'
/// [OBC]   '@' 'throw' ';'
///
StmtResult
Parser::ParseStatementOrDeclaration(StmtVector &Stmts, bool OnlyStatement,
                                    SourceLocation *TrailingElseLoc) {

  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  ParsedAttributesWithRange Attrs(AttrFactory);
  MaybeParseCXX11Attributes(Attrs, nullptr, /*MightBeObjCMessageSend*/ true);

  StmtResult Res = ParseStatementOrDeclarationAfterAttributes(Stmts,
                                 OnlyStatement, TrailingElseLoc, Attrs);

  assert((Attrs.empty() || Res.isInvalid() || Res.isUsable()) &&
         "attributes on empty statement");

  if (Attrs.empty() || Res.isInvalid())
    return Res;

  return Actions.ProcessStmtAttributes(Res.get(), Attrs.getList(), Attrs.Range);
}

namespace {
class StatementFilterCCC : public CorrectionCandidateCallback {
public:
  StatementFilterCCC(Token nextTok) : NextToken(nextTok) {
    WantTypeSpecifiers = nextTok.isOneOf(tok::l_paren, tok::less, tok::l_square,
                                         tok::identifier, tok::star, tok::amp);
    WantExpressionKeywords =
        nextTok.isOneOf(tok::l_paren, tok::identifier, tok::arrow, tok::period);
    WantRemainingKeywords =
        nextTok.isOneOf(tok::l_paren, tok::semi, tok::identifier, tok::l_brace);
    WantCXXNamedCasts = false;
  }

  bool ValidateCandidate(const TypoCorrection &candidate) override {
    if (FieldDecl *FD = candidate.getCorrectionDeclAs<FieldDecl>())
      return !candidate.getCorrectionSpecifier() || isa<ObjCIvarDecl>(FD);
    if (NextToken.is(tok::equal))
      return candidate.getCorrectionDeclAs<VarDecl>();
    if (NextToken.is(tok::period) &&
        candidate.getCorrectionDeclAs<NamespaceDecl>())
      return false;
    return CorrectionCandidateCallback::ValidateCandidate(candidate);
  }

private:
  Token NextToken;
};
}

StmtResult
Parser::ParseStatementOrDeclarationAfterAttributes(StmtVector &Stmts,
          bool OnlyStatement, SourceLocation *TrailingElseLoc,
          ParsedAttributesWithRange &Attrs) {
  const char *SemiError = nullptr;
  StmtResult Res;

  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
Retry:
  tok::TokenKind Kind  = Tok.getKind();
  SourceLocation AtLoc;
  switch (Kind) {
  case tok::at: // May be a @try or @throw statement
    {
      ProhibitAttributes(Attrs); // TODO: is it correct?
      AtLoc = ConsumeToken();  // consume @
      return ParseObjCAtStatement(AtLoc);
    }

  case tok::code_completion:
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Statement);
    cutOffParsing();
    return StmtError();

  case tok::identifier: {
    Token Next = NextToken();
    if (Next.is(tok::colon)) { // C99 6.8.1: labeled-statement
      // identifier ':' statement
      return ParseLabeledStatement(Attrs);
    }

    // Look up the identifier, and typo-correct it to a keyword if it's not
    // found.
    if (Next.isNot(tok::coloncolon)) {
      // Try to limit which sets of keywords should be included in typo
      // correction based on what the next token is.
      if (TryAnnotateName(/*IsAddressOfOperand*/ false,
                          llvm::make_unique<StatementFilterCCC>(Next)) ==
          ANK_Error) {
        // Handle errors here by skipping up to the next semicolon or '}', and
        // eat the semicolon if that's what stopped us.
        SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
        if (Tok.is(tok::semi))
          ConsumeToken();
        return StmtError();
      }

      // If the identifier was typo-corrected, try again.
      if (Tok.isNot(tok::identifier))
        goto Retry;
    }

    // Fall through
  }

  default: {
    if ((getLangOpts().CPlusPlus || !OnlyStatement) && isDeclarationStatement()) {
      SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
      DeclGroupPtrTy Decl = ParseDeclaration(Declarator::BlockContext,
                                             DeclEnd, Attrs);
      return Actions.ActOnDeclStmt(Decl, DeclStart, DeclEnd);
    }

    if (Tok.is(tok::r_brace)) {
      Diag(Tok, diag::err_expected_statement);
      return StmtError();
    }

    return ParseExprStatement();
  }

  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement();
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement();

  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement();
  case tok::semi: {                 // C99 6.8.3p3: expression[opt] ';'
    bool HasLeadingEmptyMacro = Tok.hasLeadingEmptyMacro();
    return Actions.ActOnNullStmt(ConsumeToken(), HasLeadingEmptyMacro);
  }

  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    return ParseIfStatement(TrailingElseLoc);
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    return ParseSwitchStatement(TrailingElseLoc);

  case tok::kw_while:               // C99 6.8.5.1: while-statement
    return ParseWhileStatement(TrailingElseLoc);
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    Res = ParseDoStatement();
    SemiError = "do/while";
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    return ParseForStatement(TrailingElseLoc);

  case tok::kw_goto:                // C99 6.8.6.1: goto-statement
    Res = ParseGotoStatement();
    SemiError = "goto";
    break;
  case tok::kw_continue:            // C99 6.8.6.2: continue-statement
    Res = ParseContinueStatement();
    SemiError = "continue";
    break;
  case tok::kw_break:               // C99 6.8.6.3: break-statement
    Res = ParseBreakStatement();
    SemiError = "break";
    break;
  case tok::kw_return:              // C99 6.8.6.4: return-statement
    Res = ParseReturnStatement();
    SemiError = "return";
    break;
  case tok::kw_co_return:            // C++ Coroutines: co_return statement
    Res = ParseReturnStatement();
    SemiError = "co_return";
    break;

  case tok::kw_asm: {
    ProhibitAttributes(Attrs);
    bool msAsm = false;
    Res = ParseAsmStatement(msAsm);
    Res = Actions.ActOnFinishFullStmt(Res.get());
    if (msAsm) return Res;
    SemiError = "asm";
    break;
  }

  case tok::kw___if_exists:
  case tok::kw___if_not_exists:
    ProhibitAttributes(Attrs);
    ParseMicrosoftIfExistsStatement(Stmts);
    // An __if_exists block is like a compound statement, but it doesn't create
    // a new scope.
    return StmtEmpty();

  case tok::kw_try:                 // C++ 15: try-block
    return ParseCXXTryBlock();

  case tok::kw___try:
    ProhibitAttributes(Attrs); // TODO: is it correct?
    return ParseSEHTryBlock();

  case tok::kw___leave:
    Res = ParseSEHLeaveStatement();
    SemiError = "__leave";
    break;

  case tok::annot_pragma_vis:
    ProhibitAttributes(Attrs);
    HandlePragmaVisibility();
    return StmtEmpty();

  case tok::annot_pragma_pack:
    ProhibitAttributes(Attrs);
    HandlePragmaPack();
    return StmtEmpty();

  case tok::annot_pragma_msstruct:
    ProhibitAttributes(Attrs);
    HandlePragmaMSStruct();
    return StmtEmpty();

  case tok::annot_pragma_align:
    ProhibitAttributes(Attrs);
    HandlePragmaAlign();
    return StmtEmpty();

  case tok::annot_pragma_weak:
    ProhibitAttributes(Attrs);
    HandlePragmaWeak();
    return StmtEmpty();

  case tok::annot_pragma_weakalias:
    ProhibitAttributes(Attrs);
    HandlePragmaWeakAlias();
    return StmtEmpty();

  case tok::annot_pragma_redefine_extname:
    ProhibitAttributes(Attrs);
    HandlePragmaRedefineExtname();
    return StmtEmpty();

  case tok::annot_pragma_fp_contract:
    ProhibitAttributes(Attrs);
    Diag(Tok, diag::err_pragma_fp_contract_scope);
    ConsumeToken();
    return StmtError();

  case tok::annot_pragma_opencl_extension:
    ProhibitAttributes(Attrs);
    HandlePragmaOpenCLExtension();
    return StmtEmpty();

  case tok::annot_pragma_captured:
    ProhibitAttributes(Attrs);
    return HandlePragmaCaptured();

  case tok::annot_pragma_openmp:
    ProhibitAttributes(Attrs);
    return ParseOpenMPDeclarativeOrExecutableDirective(!OnlyStatement);

  case tok::annot_pragma_ms_pointers_to_members:
    ProhibitAttributes(Attrs);
    HandlePragmaMSPointersToMembers();
    return StmtEmpty();

  case tok::annot_pragma_ms_pragma:
    ProhibitAttributes(Attrs);
    HandlePragmaMSPragma();
    return StmtEmpty();

  case tok::annot_pragma_loop_hint:
    ProhibitAttributes(Attrs);
    return ParsePragmaLoopHint(Stmts, OnlyStatement, TrailingElseLoc, Attrs);
  }

  // If we reached this code, the statement must end in a semicolon.
  if (!TryConsumeToken(tok::semi) && !Res.isInvalid()) {
    // If the result was valid, then we do want to diagnose this.  Use
    // ExpectAndConsume to emit the diagnostic, even though we know it won't
    // succeed.
    ExpectAndConsume(tok::semi, diag::err_expected_semi_after_stmt, SemiError);
    // Skip until we see a } or ;, but don't eat it.
    SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
  }

  return Res;
}

/// \brief Parse an expression statement.
StmtResult Parser::ParseExprStatement() {
  // If a case keyword is missing, this is where it should be inserted.
  Token OldToken = Tok;

  // expression[opt] ';'
  ExprResult Expr(ParseExpression());
  if (Expr.isInvalid()) {
    // If the expression is invalid, skip ahead to the next semicolon or '}'.
    // Not doing this opens us up to the possibility of infinite loops if
    // ParseExpression does not consume any tokens.
    SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
    if (Tok.is(tok::semi))
      ConsumeToken();
    return Actions.ActOnExprStmtError();
  }

  if (Tok.is(tok::colon) && getCurScope()->isSwitchScope() &&
      Actions.CheckCaseExpression(Expr.get())) {
    // If a constant expression is followed by a colon inside a switch block,
    // suggest a missing case keyword.
    Diag(OldToken, diag::err_expected_case_before_expression)
      << FixItHint::CreateInsertion(OldToken.getLocation(), "case ");

    // Recover parsing as a case statement.
    return ParseCaseStatement(/*MissingCase=*/true, Expr);
  }

  // Otherwise, eat the semicolon.
  ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
  return Actions.ActOnExprStmt(Expr);
}

/// ParseSEHTryBlockCommon
///
/// seh-try-block:
///   '__try' compound-statement seh-handler
///
/// seh-handler:
///   seh-except-block
///   seh-finally-block
///
StmtResult Parser::ParseSEHTryBlock() {
  assert(Tok.is(tok::kw___try) && "Expected '__try'");
  SourceLocation TryLoc = ConsumeToken();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected) << tok::l_brace);

  StmtResult TryBlock(ParseCompoundStatement(/*isStmtExpr=*/false,
                      Scope::DeclScope | Scope::SEHTryScope));
  if(TryBlock.isInvalid())
    return TryBlock;

  StmtResult Handler;
  if (Tok.is(tok::identifier) &&
      Tok.getIdentifierInfo() == getSEHExceptKeyword()) {
    SourceLocation Loc = ConsumeToken();
    Handler = ParseSEHExceptBlock(Loc);
  } else if (Tok.is(tok::kw___finally)) {
    SourceLocation Loc = ConsumeToken();
    Handler = ParseSEHFinallyBlock(Loc);
  } else {
    return StmtError(Diag(Tok, diag::err_seh_expected_handler));
  }

  if(Handler.isInvalid())
    return Handler;

  return Actions.ActOnSEHTryBlock(false /* IsCXXTry */,
                                  TryLoc,
                                  TryBlock.get(),
                                  Handler.get());
}

/// ParseSEHExceptBlock - Handle __except
///
/// seh-except-block:
///   '__except' '(' seh-filter-expression ')' compound-statement
///
StmtResult Parser::ParseSEHExceptBlock(SourceLocation ExceptLoc) {
  PoisonIdentifierRAIIObject raii(Ident__exception_code, false),
    raii2(Ident___exception_code, false),
    raii3(Ident_GetExceptionCode, false);

  if (ExpectAndConsume(tok::l_paren))
    return StmtError();

  ParseScope ExpectScope(this, Scope::DeclScope | Scope::ControlScope |
                                   Scope::SEHExceptScope);

  if (getLangOpts().Borland) {
    Ident__exception_info->setIsPoisoned(false);
    Ident___exception_info->setIsPoisoned(false);
    Ident_GetExceptionInfo->setIsPoisoned(false);
  }

  ExprResult FilterExpr;
  {
    ParseScopeFlags FilterScope(this, getCurScope()->getFlags() |
                                          Scope::SEHFilterScope);
    FilterExpr = Actions.CorrectDelayedTyposInExpr(ParseExpression());
  }

  if (getLangOpts().Borland) {
    Ident__exception_info->setIsPoisoned(true);
    Ident___exception_info->setIsPoisoned(true);
    Ident_GetExceptionInfo->setIsPoisoned(true);
  }

  if(FilterExpr.isInvalid())
    return StmtError();

  if (ExpectAndConsume(tok::r_paren))
    return StmtError();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected) << tok::l_brace);

  StmtResult Block(ParseCompoundStatement());

  if(Block.isInvalid())
    return Block;

  return Actions.ActOnSEHExceptBlock(ExceptLoc, FilterExpr.get(), Block.get());
}

/// ParseSEHFinallyBlock - Handle __finally
///
/// seh-finally-block:
///   '__finally' compound-statement
///
StmtResult Parser::ParseSEHFinallyBlock(SourceLocation FinallyLoc) {
  PoisonIdentifierRAIIObject raii(Ident__abnormal_termination, false),
    raii2(Ident___abnormal_termination, false),
    raii3(Ident_AbnormalTermination, false);

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected) << tok::l_brace);

  ParseScope FinallyScope(this, 0);
  Actions.ActOnStartSEHFinallyBlock();

  StmtResult Block(ParseCompoundStatement());
  if(Block.isInvalid()) {
    Actions.ActOnAbortSEHFinallyBlock();
    return Block;
  }

  return Actions.ActOnFinishSEHFinallyBlock(FinallyLoc, Block.get());
}

/// Handle __leave
///
/// seh-leave-statement:
///   '__leave' ';'
///
StmtResult Parser::ParseSEHLeaveStatement() {
  SourceLocation LeaveLoc = ConsumeToken();  // eat the '__leave'.
  return Actions.ActOnSEHLeaveStmt(LeaveLoc, getCurScope());
}

/// ParseLabeledStatement - We have an identifier and a ':' after it.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///
StmtResult Parser::ParseLabeledStatement(ParsedAttributesWithRange &attrs) {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.

  assert(Tok.is(tok::colon) && "Not a label!");

  // identifier ':' statement
  SourceLocation ColonLoc = ConsumeToken();

  // Read label attributes, if present.
  StmtResult SubStmt;
  if (Tok.is(tok::kw___attribute)) {
    ParsedAttributesWithRange TempAttrs(AttrFactory);
    ParseGNUAttributes(TempAttrs);

    // In C++, GNU attributes only apply to the label if they are followed by a
    // semicolon, to disambiguate label attributes from attributes on a labeled
    // declaration.
    //
    // This doesn't quite match what GCC does; if the attribute list is empty
    // and followed by a semicolon, GCC will reject (it appears to parse the
    // attributes as part of a statement in that case). That looks like a bug.
    if (!getLangOpts().CPlusPlus || Tok.is(tok::semi))
      attrs.takeAllFrom(TempAttrs);
    else if (isDeclarationStatement()) {
      StmtVector Stmts;
      // FIXME: We should do this whether or not we have a declaration
      // statement, but that doesn't work correctly (because ProhibitAttributes
      // can't handle GNU attributes), so only call it in the one case where
      // GNU attributes are allowed.
      SubStmt = ParseStatementOrDeclarationAfterAttributes(
          Stmts, /*OnlyStmts*/ true, nullptr, TempAttrs);
      if (!TempAttrs.empty() && !SubStmt.isInvalid())
        SubStmt = Actions.ProcessStmtAttributes(
            SubStmt.get(), TempAttrs.getList(), TempAttrs.Range);
    } else {
      Diag(Tok, diag::err_expected_after) << "__attribute__" << tok::semi;
    }
  }

  // If we've not parsed a statement yet, parse one now.
  if (!SubStmt.isInvalid() && !SubStmt.isUsable())
    SubStmt = ParseStatement();

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  LabelDecl *LD = Actions.LookupOrCreateLabel(IdentTok.getIdentifierInfo(),
                                              IdentTok.getLocation());
  if (AttributeList *Attrs = attrs.getList()) {
    Actions.ProcessDeclAttributeList(Actions.CurScope, LD, Attrs);
    attrs.clear();
  }

  return Actions.ActOnLabelStmt(IdentTok.getLocation(), LD, ColonLoc,
                                SubStmt.get());
}

/// ParseCaseStatement
///       labeled-statement:
///         'case' constant-expression ':' statement
/// [GNU]   'case' constant-expression '...' constant-expression ':' statement
///
StmtResult Parser::ParseCaseStatement(bool MissingCase, ExprResult Expr) {
  assert((MissingCase || Tok.is(tok::kw_case)) && "Not a case stmt!");

  // It is very very common for code to contain many case statements recursively
  // nested, as in (but usually without indentation):
  //  case 1:
  //    case 2:
  //      case 3:
  //         case 4:
  //           case 5: etc.
  //
  // Parsing this naively works, but is both inefficient and can cause us to run
  // out of stack space in our recursive descent parser.  As a special case,
  // flatten this recursion into an iterative loop.  This is complex and gross,
  // but all the grossness is constrained to ParseCaseStatement (and some
  // weirdness in the actions), so this is just local grossness :).

  // TopLevelCase - This is the highest level we have parsed.  'case 1' in the
  // example above.
  StmtResult TopLevelCase(true);

  // DeepestParsedCaseStmt - This is the deepest statement we have parsed, which
  // gets updated each time a new case is parsed, and whose body is unset so
  // far.  When parsing 'case 4', this is the 'case 3' node.
  Stmt *DeepestParsedCaseStmt = nullptr;

  // While we have case statements, eat and stack them.
  SourceLocation ColonLoc;
  do {
    SourceLocation CaseLoc = MissingCase ? Expr.get()->getExprLoc() :
                                           ConsumeToken();  // eat the 'case'.
    ColonLoc = SourceLocation();

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteCase(getCurScope());
      cutOffParsing();
      return StmtError();
    }

    /// We don't want to treat 'case x : y' as a potential typo for 'case x::y'.
    /// Disable this form of error recovery while we're parsing the case
    /// expression.
    ColonProtectionRAIIObject ColonProtection(*this);

    ExprResult LHS;
    if (!MissingCase) {
      LHS = ParseConstantExpression();
      if (!getLangOpts().CPlusPlus11) {
        LHS = Actions.CorrectDelayedTyposInExpr(LHS, [this](class Expr *E) {
          return Actions.VerifyIntegerConstantExpression(E);
        });
      }
      if (LHS.isInvalid()) {
        // If constant-expression is parsed unsuccessfully, recover by skipping
        // current case statement (moving to the colon that ends it).
        if (SkipUntil(tok::colon, tok::r_brace, StopAtSemi | StopBeforeMatch)) {
          TryConsumeToken(tok::colon, ColonLoc);
          continue;
        }
        return StmtError();
      }
    } else {
      LHS = Expr;
      MissingCase = false;
    }

    // GNU case range extension.
    SourceLocation DotDotDotLoc;
    ExprResult RHS;
    if (TryConsumeToken(tok::ellipsis, DotDotDotLoc)) {
      Diag(DotDotDotLoc, diag::ext_gnu_case_range);
      RHS = ParseConstantExpression();
      if (RHS.isInvalid()) {
        if (SkipUntil(tok::colon, tok::r_brace, StopAtSemi | StopBeforeMatch)) {
          TryConsumeToken(tok::colon, ColonLoc);
          continue;
        }
        return StmtError();
      }
    }

    ColonProtection.restore();

    if (TryConsumeToken(tok::colon, ColonLoc)) {
    } else if (TryConsumeToken(tok::semi, ColonLoc) ||
               TryConsumeToken(tok::coloncolon, ColonLoc)) {
      // Treat "case blah;" or "case blah::" as a typo for "case blah:".
      Diag(ColonLoc, diag::err_expected_after)
          << "'case'" << tok::colon
          << FixItHint::CreateReplacement(ColonLoc, ":");
    } else {
      SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
      Diag(ExpectedLoc, diag::err_expected_after)
          << "'case'" << tok::colon
          << FixItHint::CreateInsertion(ExpectedLoc, ":");
      ColonLoc = ExpectedLoc;
    }

    StmtResult Case =
      Actions.ActOnCaseStmt(CaseLoc, LHS.get(), DotDotDotLoc,
                            RHS.get(), ColonLoc);

    // If we had a sema error parsing this case, then just ignore it and
    // continue parsing the sub-stmt.
    if (Case.isInvalid()) {
      if (TopLevelCase.isInvalid())  // No parsed case stmts.
        return ParseStatement();
      // Otherwise, just don't add it as a nested case.
    } else {
      // If this is the first case statement we parsed, it becomes TopLevelCase.
      // Otherwise we link it into the current chain.
      Stmt *NextDeepest = Case.get();
      if (TopLevelCase.isInvalid())
        TopLevelCase = Case;
      else
        Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, Case.get());
      DeepestParsedCaseStmt = NextDeepest;
    }

    // Handle all case statements.
  } while (Tok.is(tok::kw_case));

  // If we found a non-case statement, start by parsing it.
  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Nicely diagnose the common error "switch (X) { case 4: }", which is
    // not valid.  If ColonLoc doesn't point to a valid text location, there was
    // another parsing error, so avoid producing extra diagnostics.
    if (ColonLoc.isValid()) {
      SourceLocation AfterColonLoc = PP.getLocForEndOfToken(ColonLoc);
      Diag(AfterColonLoc, diag::err_label_end_of_compound_statement)
        << FixItHint::CreateInsertion(AfterColonLoc, " ;");
    }
    SubStmt = StmtError();
  }

  // Install the body into the most deeply-nested case.
  if (DeepestParsedCaseStmt) {
    // Broken sub-stmt shouldn't prevent forming the case statement properly.
    if (SubStmt.isInvalid())
      SubStmt = Actions.ActOnNullStmt(SourceLocation());
    Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, SubStmt.get());
  }

  // Return the top level parsed statement tree.
  return TopLevelCase;
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
StmtResult Parser::ParseDefaultStatement() {
  assert(Tok.is(tok::kw_default) && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  SourceLocation ColonLoc;
  if (TryConsumeToken(tok::colon, ColonLoc)) {
  } else if (TryConsumeToken(tok::semi, ColonLoc)) {
    // Treat "default;" as a typo for "default:".
    Diag(ColonLoc, diag::err_expected_after)
        << "'default'" << tok::colon
        << FixItHint::CreateReplacement(ColonLoc, ":");
  } else {
    SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
    Diag(ExpectedLoc, diag::err_expected_after)
        << "'default'" << tok::colon
        << FixItHint::CreateInsertion(ExpectedLoc, ":");
    ColonLoc = ExpectedLoc;
  }

  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Diagnose the common error "switch (X) {... default: }", which is
    // not valid.
    SourceLocation AfterColonLoc = PP.getLocForEndOfToken(ColonLoc);
    Diag(AfterColonLoc, diag::err_label_end_of_compound_statement)
      << FixItHint::CreateInsertion(AfterColonLoc, " ;");
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);

  return Actions.ActOnDefaultStmt(DefaultLoc, ColonLoc,
                                  SubStmt.get(), getCurScope());
}

StmtResult Parser::ParseCompoundStatement(bool isStmtExpr) {
  return ParseCompoundStatement(isStmtExpr, Scope::DeclScope);
}

/// ParseCompoundStatement - Parse a "{}" block.
///
///       compound-statement: [C99 6.8.2]
///         { block-item-list[opt] }
/// [GNU]   { label-declarations block-item-list } [TODO]
///
///       block-item-list:
///         block-item
///         block-item-list block-item
///
///       block-item:
///         declaration
/// [GNU]   '__extension__' declaration
///         statement
///
/// [GNU] label-declarations:
/// [GNU]   label-declaration
/// [GNU]   label-declarations label-declaration
///
/// [GNU] label-declaration:
/// [GNU]   '__label__' identifier-list ';'
///
StmtResult Parser::ParseCompoundStatement(bool isStmtExpr,
                                          unsigned ScopeFlags) {
  assert(Tok.is(tok::l_brace) && "Not a compount stmt!");

  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  ParseScope CompoundScope(this, ScopeFlags);

  // Parse the statements in the body.
  return ParseCompoundStatementBody(isStmtExpr);
}

/// Parse any pragmas at the start of the compound expression. We handle these
/// separately since some pragmas (FP_CONTRACT) must appear before any C
/// statement in the compound, but may be intermingled with other pragmas.
void Parser::ParseCompoundStatementLeadingPragmas() {
  bool checkForPragmas = true;
  while (checkForPragmas) {
    switch (Tok.getKind()) {
    case tok::annot_pragma_vis:
      HandlePragmaVisibility();
      break;
    case tok::annot_pragma_pack:
      HandlePragmaPack();
      break;
    case tok::annot_pragma_msstruct:
      HandlePragmaMSStruct();
      break;
    case tok::annot_pragma_align:
      HandlePragmaAlign();
      break;
    case tok::annot_pragma_weak:
      HandlePragmaWeak();
      break;
    case tok::annot_pragma_weakalias:
      HandlePragmaWeakAlias();
      break;
    case tok::annot_pragma_redefine_extname:
      HandlePragmaRedefineExtname();
      break;
    case tok::annot_pragma_opencl_extension:
      HandlePragmaOpenCLExtension();
      break;
    case tok::annot_pragma_fp_contract:
      HandlePragmaFPContract();
      break;
    case tok::annot_pragma_ms_pointers_to_members:
      HandlePragmaMSPointersToMembers();
      break;
    case tok::annot_pragma_ms_pragma:
      HandlePragmaMSPragma();
      break;
    default:
      checkForPragmas = false;
      break;
    }
  }

}

/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ActOnCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
StmtResult Parser::ParseCompoundStatementBody(bool isStmtExpr) {
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
                                Tok.getLocation(),
                                "in compound statement ('{}')");

  // Record the state of the FP_CONTRACT pragma, restore on leaving the
  // compound statement.
  Sema::FPContractStateRAII SaveFPContractState(Actions);

  InMessageExpressionRAIIObject InMessage(*this, false);
  BalancedDelimiterTracker T(*this, tok::l_brace);
  if (T.consumeOpen())
    return StmtError();

  Sema::CompoundScopeRAII CompoundScope(Actions);

  // Parse any pragmas at the beginning of the compound statement.
  ParseCompoundStatementLeadingPragmas();

  StmtVector Stmts;

  // "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt regardless of the language.
  while (Tok.is(tok::kw___label__)) {
    SourceLocation LabelLoc = ConsumeToken();

    SmallVector<Decl *, 8> DeclsInGroup;
    while (1) {
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected) << tok::identifier;
        break;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      SourceLocation IdLoc = ConsumeToken();
      DeclsInGroup.push_back(Actions.LookupOrCreateLabel(II, IdLoc, LabelLoc));

      if (!TryConsumeToken(tok::comma))
        break;
    }

    DeclSpec DS(AttrFactory);
    DeclGroupPtrTy Res =
        Actions.FinalizeDeclaratorGroup(getCurScope(), DS, DeclsInGroup);
    StmtResult R = Actions.ActOnDeclStmt(Res, LabelLoc, Tok.getLocation());

    ExpectAndConsumeSemi(diag::err_expected_semi_declaration);
    if (R.isUsable())
      Stmts.push_back(R.get());
  }

  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof) &&
         !tryParseMisplacedModuleImport()) {
    if (Tok.is(tok::annot_pragma_unused)) {
      HandlePragmaUnused();
      continue;
    }

    StmtResult R;
    if (Tok.isNot(tok::kw___extension__)) {
      R = ParseStatementOrDeclaration(Stmts, false);
    } else {
      // __extension__ can start declarations and it can also be a unary
      // operator for expressions.  Consume multiple __extension__ markers here
      // until we can determine which is which.
      // FIXME: This loses extension expressions in the AST!
      SourceLocation ExtLoc = ConsumeToken();
      while (Tok.is(tok::kw___extension__))
        ConsumeToken();

      ParsedAttributesWithRange attrs(AttrFactory);
      MaybeParseCXX11Attributes(attrs, nullptr,
                                /*MightBeObjCMessageSend*/ true);

      // If this is the start of a declaration, parse it as such.
      if (isDeclarationStatement()) {
        // __extension__ silences extension warnings in the subdeclaration.
        // FIXME: Save the __extension__ on the decl as a node somehow?
        ExtensionRAIIObject O(Diags);

        SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
        DeclGroupPtrTy Res = ParseDeclaration(Declarator::BlockContext, DeclEnd,
                                              attrs);
        R = Actions.ActOnDeclStmt(Res, DeclStart, DeclEnd);
      } else {
        // Otherwise this was a unary __extension__ marker.
        ExprResult Res(ParseExpressionWithLeadingExtension(ExtLoc));

        if (Res.isInvalid()) {
          SkipUntil(tok::semi);
          continue;
        }

        // FIXME: Use attributes?
        // Eat the semicolon at the end of stmt and convert the expr into a
        // statement.
        ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
        R = Actions.ActOnExprStmt(Res);
      }
    }

    if (R.isUsable())
      Stmts.push_back(R.get());
  }

  SourceLocation CloseLoc = Tok.getLocation();

  // We broke out of the while loop because we found a '}' or EOF.
  if (!T.consumeClose())
    // Recover by creating a compound statement with what we parsed so far,
    // instead of dropping everything and returning StmtError();
    CloseLoc = T.getCloseLocation();

  return Actions.ActOnCompoundStmt(T.getOpenLocation(), CloseLoc,
                                   Stmts, isStmtExpr);
}

/// ParseParenExprOrCondition:
/// [C  ]     '(' expression ')'
/// [C++]     '(' condition ')'       [not allowed if OnlyAllowCondition=true]
///
/// This function parses and performs error recovery on the specified condition
/// or expression (depending on whether we're in C++ or C mode).  This function
/// goes out of its way to recover well.  It returns true if there was a parser
/// error (the right paren couldn't be found), which indicates that the caller
/// should try to recover harder.  It returns false if the condition is
/// successfully parsed.  Note that a successful parse can still have semantic
/// errors in the condition.
bool Parser::ParseParenExprOrCondition(ExprResult &ExprResult,
                                       Decl *&DeclResult,
                                       SourceLocation Loc,
                                       bool ConvertToBoolean) {
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  if (getLangOpts().CPlusPlus)
    ParseCXXCondition(ExprResult, DeclResult, Loc, ConvertToBoolean);
  else {
    ExprResult = ParseExpression();
    DeclResult = nullptr;

    // If required, convert to a boolean value.
    if (!ExprResult.isInvalid() && ConvertToBoolean)
      ExprResult
        = Actions.ActOnBooleanCondition(getCurScope(), Loc, ExprResult.get());
  }

  // If the parser was confused by the condition and we don't have a ')', try to
  // recover by skipping ahead to a semi and bailing out.  If condexp is
  // semantically invalid but we have well formed code, keep going.
  if (ExprResult.isInvalid() && !DeclResult && Tok.isNot(tok::r_paren)) {
    SkipUntil(tok::semi);
    // Skipping may have stopped if it found the containing ')'.  If so, we can
    // continue parsing the if statement.
    if (Tok.isNot(tok::r_paren))
      return true;
  }

  // Otherwise the condition is valid or the rparen is present.
  T.consumeClose();

  // Check for extraneous ')'s to catch things like "if (foo())) {".  We know
  // that all callers are looking for a statement after the condition, so ")"
  // isn't valid.
  while (Tok.is(tok::r_paren)) {
    Diag(Tok, diag::err_extraneous_rparen_in_condition)
      << FixItHint::CreateRemoval(Tok.getLocation());
    ConsumeParen();
  }

  return false;
}


/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
/// [C++]   'if' '(' condition ')' statement
/// [C++]   'if' '(' condition ')' statement 'else' statement
///
StmtResult Parser::ParseIfStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "if";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.4p3 - In C99, the if statement is a block.  This is not
  // the case for C90.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  ParseScope IfScope(this, Scope::DeclScope | Scope::ControlScope, C99orCXX);

  // Parse the condition.
  ExprResult CondExp;
  Decl *CondVar = nullptr;
  if (ParseParenExprOrCondition(CondExp, CondVar, IfLoc, true))
    return StmtError();

  FullExprArg FullCondExp(Actions.MakeFullExpr(CondExp.get(), IfLoc));

  // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.4p1:
  // The substatement in a selection-statement (each substatement, in the else
  // form of the if statement) implicitly defines a local scope.
  //
  // For C++ we create a scope for the condition and a new scope for
  // substatements because:
  // -When the 'then' scope exits, we want the condition declaration to still be
  //    active for the 'else' scope too.
  // -Sema will detect name clashes by considering declarations of a
  //    'ControlScope' as part of its direct subscope.
  // -If we wanted the condition and substatement to be in the same scope, we
  //    would have to notify ParseStatement not to create a new scope. It's
  //    simpler to let it create a new scope.
  //
  ParseScope InnerScope(this, Scope::DeclScope, C99orCXX, Tok.is(tok::l_brace));

  // Read the 'then' stmt.
  SourceLocation ThenStmtLoc = Tok.getLocation();

  SourceLocation InnerStatementTrailingElseLoc;
  StmtResult ThenStmt(ParseStatement(&InnerStatementTrailingElseLoc));

  // Pop the 'if' scope if needed.
  InnerScope.Exit();

  // If it has an else, parse it.
  SourceLocation ElseLoc;
  SourceLocation ElseStmtLoc;
  StmtResult ElseStmt;

  if (Tok.is(tok::kw_else)) {
    if (TrailingElseLoc)
      *TrailingElseLoc = Tok.getLocation();

    ElseLoc = ConsumeToken();
    ElseStmtLoc = Tok.getLocation();

    // C99 6.8.4p3 - In C99, the body of the if statement is a scope, even if
    // there is no compound stmt.  C90 does not have this clause.  We only do
    // this if the body isn't a compound statement to avoid push/pop in common
    // cases.
    //
    // C++ 6.4p1:
    // The substatement in a selection-statement (each substatement, in the else
    // form of the if statement) implicitly defines a local scope.
    //
    ParseScope InnerScope(this, Scope::DeclScope, C99orCXX, Tok.is(tok::l_brace));

    ElseStmt = ParseStatement();

    // Pop the 'else' scope if needed.
    InnerScope.Exit();
  } else if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteAfterIf(getCurScope());
    cutOffParsing();
    return StmtError();
  } else if (InnerStatementTrailingElseLoc.isValid()) {
    Diag(InnerStatementTrailingElseLoc, diag::warn_dangling_else);
  }

  IfScope.Exit();

  // If the then or else stmt is invalid and the other is valid (and present),
  // make turn the invalid one into a null stmt to avoid dropping the other
  // part.  If both are invalid, return error.
  if ((ThenStmt.isInvalid() && ElseStmt.isInvalid()) ||
      (ThenStmt.isInvalid() && ElseStmt.get() == nullptr) ||
      (ThenStmt.get() == nullptr && ElseStmt.isInvalid())) {
    // Both invalid, or one is invalid and other is non-present: return error.
    return StmtError();
  }

  // Now if either are invalid, replace with a ';'.
  if (ThenStmt.isInvalid())
    ThenStmt = Actions.ActOnNullStmt(ThenStmtLoc);
  if (ElseStmt.isInvalid())
    ElseStmt = Actions.ActOnNullStmt(ElseStmtLoc);

  return Actions.ActOnIfStmt(IfLoc, FullCondExp, CondVar, ThenStmt.get(),
                             ElseLoc, ElseStmt.get());
}

/// ParseSwitchStatement
///       switch-statement:
///         'switch' '(' expression ')' statement
/// [C++]   'switch' '(' condition ')' statement
StmtResult Parser::ParseSwitchStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "switch";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.4p3 - In C99, the switch statement is a block.  This is
  // not the case for C90.  Start the switch scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  unsigned ScopeFlags = Scope::SwitchScope;
  if (C99orCXX)
    ScopeFlags |= Scope::DeclScope | Scope::ControlScope;
  ParseScope SwitchScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = nullptr;
  if (ParseParenExprOrCondition(Cond, CondVar, SwitchLoc, false))
    return StmtError();

  StmtResult Switch
    = Actions.ActOnStartOfSwitchStmt(SwitchLoc, Cond.get(), CondVar);

  if (Switch.isInvalid()) {
    // Skip the switch body.
    // FIXME: This is not optimal recovery, but parsing the body is more
    // dangerous due to the presence of case and default statements, which
    // will have no place to connect back with the switch.
    if (Tok.is(tok::l_brace)) {
      ConsumeBrace();
      SkipUntil(tok::r_brace);
    } else
      SkipUntil(tok::semi);
    return Switch;
  }

  // C99 6.8.4p3 - In C99, the body of the switch statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.4p1:
  // The substatement in a selection-statement (each substatement, in the else
  // form of the if statement) implicitly defines a local scope.
  //
  // See comments in ParseIfStatement for why we create a scope for the
  // condition and a new scope for substatement in C++.
  //
  getCurScope()->AddFlags(Scope::BreakScope);
  ParseScope InnerScope(this, Scope::DeclScope, C99orCXX, Tok.is(tok::l_brace));

  // We have incremented the mangling number for the SwitchScope and the
  // InnerScope, which is one too many.
  if (C99orCXX)
    getCurScope()->decrementMSManglingNumber();

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the scopes.
  InnerScope.Exit();
  SwitchScope.Exit();

  return Actions.ActOnFinishSwitchStmt(SwitchLoc, Switch.get(), Body.get());
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
/// [C++]   'while' '(' condition ')' statement
StmtResult Parser::ParseWhileStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "while";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;

  // C99 6.8.5p5 - In C99, the while statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  //
  unsigned ScopeFlags;
  if (C99orCXX)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
                 Scope::DeclScope  | Scope::ControlScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;
  ParseScope WhileScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = nullptr;
  if (ParseParenExprOrCondition(Cond, CondVar, WhileLoc, true))
    return StmtError();

  FullExprArg FullCond(Actions.MakeFullExpr(Cond.get(), WhileLoc));

  // C99 6.8.5p5 - In C99, the body of the while statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for the
  // condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope, C99orCXX, Tok.is(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();
  WhileScope.Exit();

  if ((Cond.isInvalid() && !CondVar) || Body.isInvalid())
    return StmtError();

  return Actions.ActOnWhileStmt(WhileLoc, FullCond, CondVar, Body.get());
}

/// ParseDoStatement
///       do-statement: [C99 6.8.5.2]
///         'do' statement 'while' '(' expression ')' ';'
/// Note: this lets the caller parse the end ';'.
StmtResult Parser::ParseDoStatement() {
  assert(Tok.is(tok::kw_do) && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.

  // C99 6.8.5p5 - In C99, the do statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  unsigned ScopeFlags;
  if (getLangOpts().C99)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope DoScope(this, ScopeFlags);

  // C99 6.8.5p5 - In C99, the body of the do statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause. We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  bool C99orCXX = getLangOpts().C99 || getLangOpts().CPlusPlus;
  ParseScope InnerScope(this, Scope::DeclScope, C99orCXX, Tok.is(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  if (Tok.isNot(tok::kw_while)) {
    if (!Body.isInvalid()) {
      Diag(Tok, diag::err_expected_while);
      Diag(DoLoc, diag::note_matching) << "'do'";
      SkipUntil(tok::semi, StopBeforeMatch);
    }
    return StmtError();
  }
  SourceLocation WhileLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "do/while";
    SkipUntil(tok::semi, StopBeforeMatch);
    return StmtError();
  }

  // Parse the parenthesized expression.
  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  // A do-while expression is not a condition, so can't have attributes.
  DiagnoseAndSkipCXX11Attributes();

  ExprResult Cond = ParseExpression();
  T.consumeClose();
  DoScope.Exit();

  if (Cond.isInvalid() || Body.isInvalid())
    return StmtError();

  return Actions.ActOnDoStmt(DoLoc, Body.get(), WhileLoc, T.getOpenLocation(),
                             Cond.get(), T.getCloseLocation());
}

bool Parser::isForRangeIdentifier() {
  assert(Tok.is(tok::identifier));

  const Token &Next = NextToken();
  if (Next.is(tok::colon))
    return true;

  if (Next.isOneOf(tok::l_square, tok::kw_alignas)) {
    TentativeParsingAction PA(*this);
    ConsumeToken();
    SkipCXX11Attributes();
    bool Result = Tok.is(tok::colon);
    PA.Revert();
    return Result;
  }

  return false;
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
/// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
/// [C++]       statement
/// [C++0x] 'for'
///             'co_await'[opt]    [Coroutines]
///             '(' for-range-declaration ':' for-range-initializer ')'
///             statement
/// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
/// [OBJC2] 'for' '(' expr 'in' expr ')' statement
///
/// [C++] for-init-statement:
/// [C++]   expression-statement
/// [C++]   simple-declaration
///
/// [C++0x] for-range-declaration:
/// [C++0x]   attribute-specifier-seq[opt] type-specifier-seq declarator
/// [C++0x] for-range-initializer:
/// [C++0x]   expression
/// [C++0x]   braced-init-list            [TODO]
StmtResult Parser::ParseForStatement(SourceLocation *TrailingElseLoc) {
  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.

  SourceLocation CoawaitLoc;
  if (Tok.is(tok::kw_co_await))
    CoawaitLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "for";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXXorObjC = getLangOpts().C99 || getLangOpts().CPlusPlus ||
    getLangOpts().ObjC1;

  // C99 6.8.5p5 - In C99, the for statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  //
  // C++ 6.4p3:
  // A name introduced by a declaration in a condition is in scope from its
  // point of declaration until the end of the substatements controlled by the
  // condition.
  // C++ 3.3.2p4:
  // Names declared in the for-init-statement, and in the condition of if,
  // while, for, and switch statements are local to the if, while, for, or
  // switch statement (including the controlled statement).
  // C++ 6.5.3p1:
  // Names declared in the for-init-statement are in the same declarative-region
  // as those declared in the condition.
  //
  unsigned ScopeFlags = 0;
  if (C99orCXXorObjC)
    ScopeFlags = Scope::DeclScope | Scope::ControlScope;

  ParseScope ForScope(this, ScopeFlags);

  BalancedDelimiterTracker T(*this, tok::l_paren);
  T.consumeOpen();

  ExprResult Value;

  bool ForEach = false, ForRange = false;
  StmtResult FirstPart;
  bool SecondPartIsInvalid = false;
  FullExprArg SecondPart(Actions);
  ExprResult Collection;
  ForRangeInit ForRangeInit;
  FullExprArg ThirdPart(Actions);
  Decl *SecondVar = nullptr;

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(),
                                     C99orCXXorObjC? Sema::PCC_ForInit
                                                   : Sema::PCC_Expression);
    cutOffParsing();
    return StmtError();
  }

  ParsedAttributesWithRange attrs(AttrFactory);
  MaybeParseCXX11Attributes(attrs);

  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    ProhibitAttributes(attrs);
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (getLangOpts().CPlusPlus && Tok.is(tok::identifier) &&
             isForRangeIdentifier()) {
    ProhibitAttributes(attrs);
    IdentifierInfo *Name = Tok.getIdentifierInfo();
    SourceLocation Loc = ConsumeToken();
    MaybeParseCXX11Attributes(attrs);

    ForRangeInit.ColonLoc = ConsumeToken();
    if (Tok.is(tok::l_brace))
      ForRangeInit.RangeExpr = ParseBraceInitializer();
    else
      ForRangeInit.RangeExpr = ParseExpression();

    Diag(Loc, diag::err_for_range_identifier)
      << ((getLangOpts().CPlusPlus11 && !getLangOpts().CPlusPlus1z)
              ? FixItHint::CreateInsertion(Loc, "auto &&")
              : FixItHint());

    FirstPart = Actions.ActOnCXXForRangeIdentifier(getCurScope(), Loc, Name,
                                                   attrs, attrs.Range.getEnd());
    ForRange = true;
  } else if (isForInitDeclaration()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!C99orCXXorObjC)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);

    // In C++0x, "for (T NS:a" might not be a typo for ::
    bool MightBeForRangeStmt = getLangOpts().CPlusPlus;
    ColonProtectionRAIIObject ColonProtection(*this, MightBeForRangeStmt);

    SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
    DeclGroupPtrTy DG = ParseSimpleDeclaration(
        Declarator::ForContext, DeclEnd, attrs, false,
        MightBeForRangeStmt ? &ForRangeInit : nullptr);
    FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());
    if (ForRangeInit.ParsedForRangeDecl()) {
      Diag(ForRangeInit.ColonLoc, getLangOpts().CPlusPlus11 ?
           diag::warn_cxx98_compat_for_range : diag::ext_for_range);

      ForRange = true;
    } else if (Tok.is(tok::semi)) {  // for (int x = 4;
      ConsumeToken();
    } else if ((ForEach = isTokIdentifier_in())) {
      Actions.ActOnForEachDeclStmt(DG);
      // ObjC: for (id x in expr)
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DG);
        cutOffParsing();
        return StmtError();
      }
      Collection = ParseExpression();
    } else {
      Diag(Tok, diag::err_expected_semi_for);
    }
  } else {
    ProhibitAttributes(attrs);
    Value = Actions.CorrectDelayedTyposInExpr(ParseExpression());

    ForEach = isTokIdentifier_in();

    // Turn the expression into a stmt.
    if (!Value.isInvalid()) {
      if (ForEach)
        FirstPart = Actions.ActOnForEachLValueExpr(Value.get());
      else
        FirstPart = Actions.ActOnExprStmt(Value);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (ForEach) {
      ConsumeToken(); // consume 'in'

      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DeclGroupPtrTy());
        cutOffParsing();
        return StmtError();
      }
      Collection = ParseExpression();
    } else if (getLangOpts().CPlusPlus11 && Tok.is(tok::colon) && FirstPart.get()) {
      // User tried to write the reasonable, but ill-formed, for-range-statement
      //   for (expr : expr) { ... }
      Diag(Tok, diag::err_for_range_expected_decl)
        << FirstPart.get()->getSourceRange();
      SkipUntil(tok::r_paren, StopBeforeMatch);
      SecondPartIsInvalid = true;
    } else {
      if (!Value.isInvalid()) {
        Diag(Tok, diag::err_expected_semi_for);
      } else {
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, StopAtSemi | StopBeforeMatch);
        if (Tok.is(tok::semi))
          ConsumeToken();
      }
    }
  }

  // Parse the second part of the for specifier.
  getCurScope()->AddFlags(Scope::BreakScope | Scope::ContinueScope);
  if (!ForEach && !ForRange) {
    assert(!SecondPart.get() && "Shouldn't have a second expression yet.");
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) {  // for (...;;
      // no second part.
    } else if (Tok.is(tok::r_paren)) {
      // missing both semicolons.
    } else {
      ExprResult Second;
      if (getLangOpts().CPlusPlus)
        ParseCXXCondition(Second, SecondVar, ForLoc, true);
      else {
        Second = ParseExpression();
        if (!Second.isInvalid())
          Second = Actions.ActOnBooleanCondition(getCurScope(), ForLoc,
                                                 Second.get());
      }
      SecondPartIsInvalid = Second.isInvalid();
      SecondPart = Actions.MakeFullExpr(Second.get(), ForLoc);
    }

    if (Tok.isNot(tok::semi)) {
      if (!SecondPartIsInvalid || SecondVar)
        Diag(Tok, diag::err_expected_semi_for);
      else
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, StopAtSemi | StopBeforeMatch);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    }

    // Parse the third part of the for specifier.
    if (Tok.isNot(tok::r_paren)) {   // for (...;...;)
      ExprResult Third = ParseExpression();
      // FIXME: The C++11 standard doesn't actually say that this is a
      // discarded-value expression, but it clearly should be.
      ThirdPart = Actions.MakeFullDiscardedValueExpr(Third.get());
    }
  }
  // Match the ')'.
  T.consumeClose();

  // C++ Coroutines [stmt.iter]:
  //   'co_await' can only be used for a range-based for statement.
  if (CoawaitLoc.isValid() && !ForRange) {
    Diag(CoawaitLoc, diag::err_for_co_await_not_range_for);
    CoawaitLoc = SourceLocation();
  }

  // We need to perform most of the semantic analysis for a C++0x for-range
  // statememt before parsing the body, in order to be able to deduce the type
  // of an auto-typed loop variable.
  StmtResult ForRangeStmt;
  StmtResult ForEachStmt;

  if (ForRange) {
    ForRangeStmt = Actions.ActOnCXXForRangeStmt(
        getCurScope(), ForLoc, CoawaitLoc, FirstPart.get(),
        ForRangeInit.ColonLoc, ForRangeInit.RangeExpr.get(),
        T.getCloseLocation(), Sema::BFRK_Build);

  // Similarly, we need to do the semantic analysis for a for-range
  // statement immediately in order to close over temporaries correctly.
  } else if (ForEach) {
    ForEachStmt = Actions.ActOnObjCForCollectionStmt(ForLoc,
                                                     FirstPart.get(),
                                                     Collection.get(),
                                                     T.getCloseLocation());
  } else {
    // In OpenMP loop region loop control variable must be captured and be
    // private. Perform analysis of first part (if any).
    if (getLangOpts().OpenMP && FirstPart.isUsable()) {
      Actions.ActOnOpenMPLoopInitialization(ForLoc, FirstPart.get());
    }
  }

  // C99 6.8.5p5 - In C99, the body of the for statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause.  We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  // See comments in ParseIfStatement for why we create a scope for
  // for-init-statement/condition and a new scope for substatement in C++.
  //
  ParseScope InnerScope(this, Scope::DeclScope, C99orCXXorObjC,
                        Tok.is(tok::l_brace));

  // The body of the for loop has the same local mangling number as the
  // for-init-statement.
  // It will only be incremented if the body contains other things that would
  // normally increment the mangling number (like a compound statement).
  if (C99orCXXorObjC)
    getCurScope()->decrementMSManglingNumber();

  // Read the body statement.
  StmtResult Body(ParseStatement(TrailingElseLoc));

  // Pop the body scope if needed.
  InnerScope.Exit();

  // Leave the for-scope.
  ForScope.Exit();

  if (Body.isInvalid())
    return StmtError();

  if (ForEach)
   return Actions.FinishObjCForCollectionStmt(ForEachStmt.get(),
                                              Body.get());

  if (ForRange)
    return Actions.FinishCXXForRangeStmt(ForRangeStmt.get(), Body.get());

  return Actions.ActOnForStmt(ForLoc, T.getOpenLocation(), FirstPart.get(),
                              SecondPart, SecondVar, ThirdPart,
                              T.getCloseLocation(), Body.get());
}

/// ParseGotoStatement
///       jump-statement:
///         'goto' identifier ';'
/// [GNU]   'goto' '*' expression ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseGotoStatement() {
  assert(Tok.is(tok::kw_goto) && "Not a goto stmt!");
  SourceLocation GotoLoc = ConsumeToken();  // eat the 'goto'.

  StmtResult Res;
  if (Tok.is(tok::identifier)) {
    LabelDecl *LD = Actions.LookupOrCreateLabel(Tok.getIdentifierInfo(),
                                                Tok.getLocation());
    Res = Actions.ActOnGotoStmt(GotoLoc, Tok.getLocation(), LD);
    ConsumeToken();
  } else if (Tok.is(tok::star)) {
    // GNU indirect goto extension.
    Diag(Tok, diag::ext_gnu_indirect_goto);
    SourceLocation StarLoc = ConsumeToken();
    ExprResult R(ParseExpression());
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, StopBeforeMatch);
      return StmtError();
    }
    Res = Actions.ActOnIndirectGotoStmt(GotoLoc, StarLoc, R.get());
  } else {
    Diag(Tok, diag::err_expected) << tok::identifier;
    return StmtError();
  }

  return Res;
}

/// ParseContinueStatement
///       jump-statement:
///         'continue' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseContinueStatement() {
  SourceLocation ContinueLoc = ConsumeToken();  // eat the 'continue'.
  return Actions.ActOnContinueStmt(ContinueLoc, getCurScope());
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseBreakStatement() {
  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ActOnBreakStmt(BreakLoc, getCurScope());
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
///         'return' braced-init-list ';'
///         'co_return' expression[opt] ';'
///         'co_return' braced-init-list ';'
StmtResult Parser::ParseReturnStatement() {
  assert((Tok.is(tok::kw_return) || Tok.is(tok::kw_co_return)) &&
         "Not a return stmt!");
  bool IsCoreturn = Tok.is(tok::kw_co_return);
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.

  ExprResult R;
  if (Tok.isNot(tok::semi)) {
    // FIXME: Code completion for co_return.
    if (Tok.is(tok::code_completion) && !IsCoreturn) {
      Actions.CodeCompleteReturn(getCurScope());
      cutOffParsing();
      return StmtError();
    }

    if (Tok.is(tok::l_brace) && getLangOpts().CPlusPlus) {
      R = ParseInitializer();
      if (R.isUsable())
        Diag(R.get()->getLocStart(), getLangOpts().CPlusPlus11 ?
             diag::warn_cxx98_compat_generalized_initializer_lists :
             diag::ext_generalized_initializer_lists)
          << R.get()->getSourceRange();
    } else
      R = ParseExpression();
    if (R.isInvalid()) {
      SkipUntil(tok::r_brace, StopAtSemi | StopBeforeMatch);
      return StmtError();
    }
  }
  if (IsCoreturn)
    return Actions.ActOnCoreturnStmt(ReturnLoc, R.get());
  return Actions.ActOnReturnStmt(ReturnLoc, R.get(), getCurScope());
}

StmtResult Parser::ParsePragmaLoopHint(StmtVector &Stmts, bool OnlyStatement,
                                       SourceLocation *TrailingElseLoc,
                                       ParsedAttributesWithRange &Attrs) {
  // Create temporary attribute list.
  ParsedAttributesWithRange TempAttrs(AttrFactory);

  // Get loop hints and consume annotated token.
  while (Tok.is(tok::annot_pragma_loop_hint)) {
    LoopHint Hint;
    if (!HandlePragmaLoopHint(Hint))
      continue;

    ArgsUnion ArgHints[] = {Hint.PragmaNameLoc, Hint.OptionLoc, Hint.StateLoc,
                            ArgsUnion(Hint.ValueExpr)};
    TempAttrs.addNew(Hint.PragmaNameLoc->Ident, Hint.Range, nullptr,
                     Hint.PragmaNameLoc->Loc, ArgHints, 4,
                     AttributeList::AS_Pragma);
  }

  // Get the next statement.
  MaybeParseCXX11Attributes(Attrs);

  StmtResult S = ParseStatementOrDeclarationAfterAttributes(
      Stmts, OnlyStatement, TrailingElseLoc, Attrs);

  Attrs.takeAllFrom(TempAttrs);
  return S;
}

Decl *Parser::ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();

  if (SkipFunctionBodies && (!Decl || Actions.canSkipFunctionBody(Decl)) &&
      trySkippingFunctionBody()) {
    BodyScope.Exit();
    return Actions.ActOnSkippedFunctionBody(Decl);
  }

  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, LBraceLoc,
                                      "parsing function body");

  // Do not enter a scope for the brace, as the arguments are in the same scope
  // (the function body) as the body itself.  Instead, just read the statement
  // list and put it into a CompoundStmt for safe keeping.
  StmtResult FnBody(ParseCompoundStatementBody());

  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid()) {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc, None, false);
  }

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.get());
}

/// ParseFunctionTryBlock - Parse a C++ function-try-block.
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
Decl *Parser::ParseFunctionTryBlock(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");
  SourceLocation TryLoc = ConsumeToken();

  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, TryLoc,
                                      "parsing function try block");

  // Constructor initializer list?
  if (Tok.is(tok::colon))
    ParseConstructorInitializer(Decl);
  else
    Actions.ActOnDefaultCtorInitializers(Decl);

  if (SkipFunctionBodies && Actions.canSkipFunctionBody(Decl) &&
      trySkippingFunctionBody()) {
    BodyScope.Exit();
    return Actions.ActOnSkippedFunctionBody(Decl);
  }

  SourceLocation LBraceLoc = Tok.getLocation();
  StmtResult FnBody(ParseCXXTryBlockCommon(TryLoc, /*FnTry*/true));
  // If we failed to parse the try-catch, we just give the function an empty
  // compound statement as the body.
  if (FnBody.isInvalid()) {
    Sema::CompoundScopeRAII CompoundScope(Actions);
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc, None, false);
  }

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.get());
}

bool Parser::trySkippingFunctionBody() {
  assert(Tok.is(tok::l_brace));
  assert(SkipFunctionBodies &&
         "Should only be called when SkipFunctionBodies is enabled");

  if (!PP.isCodeCompletionEnabled()) {
    ConsumeBrace();
    SkipUntil(tok::r_brace);
    return true;
  }

  // We're in code-completion mode. Skip parsing for all function bodies unless
  // the body contains the code-completion point.
  TentativeParsingAction PA(*this);
  ConsumeBrace();
  if (SkipUntil(tok::r_brace, StopAtCodeCompletion)) {
    PA.Commit();
    return true;
  }

  PA.Revert();
  return false;
}

/// ParseCXXTryBlock - Parse a C++ try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
StmtResult Parser::ParseCXXTryBlock() {
  assert(Tok.is(tok::kw_try) && "Expected 'try'");

  SourceLocation TryLoc = ConsumeToken();
  return ParseCXXTryBlockCommon(TryLoc);
}

/// ParseCXXTryBlockCommon - Parse the common part of try-block and
/// function-try-block.
///
///       try-block:
///         'try' compound-statement handler-seq
///
///       function-try-block:
///         'try' ctor-initializer[opt] compound-statement handler-seq
///
///       handler-seq:
///         handler handler-seq[opt]
///
///       [Borland] try-block:
///         'try' compound-statement seh-except-block
///         'try' compound-statement seh-finally-block
///
StmtResult Parser::ParseCXXTryBlockCommon(SourceLocation TryLoc, bool FnTry) {
  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected) << tok::l_brace);

  StmtResult TryBlock(ParseCompoundStatement(/*isStmtExpr=*/false,
                      Scope::DeclScope | Scope::TryScope |
                        (FnTry ? Scope::FnTryCatchScope : 0)));
  if (TryBlock.isInvalid())
    return TryBlock;

  // Borland allows SEH-handlers with 'try'

  if ((Tok.is(tok::identifier) &&
       Tok.getIdentifierInfo() == getSEHExceptKeyword()) ||
      Tok.is(tok::kw___finally)) {
    // TODO: Factor into common return ParseSEHHandlerCommon(...)
    StmtResult Handler;
    if(Tok.getIdentifierInfo() == getSEHExceptKeyword()) {
      SourceLocation Loc = ConsumeToken();
      Handler = ParseSEHExceptBlock(Loc);
    }
    else {
      SourceLocation Loc = ConsumeToken();
      Handler = ParseSEHFinallyBlock(Loc);
    }
    if(Handler.isInvalid())
      return Handler;

    return Actions.ActOnSEHTryBlock(true /* IsCXXTry */,
                                    TryLoc,
                                    TryBlock.get(),
                                    Handler.get());
  }
  else {
    StmtVector Handlers;

    // C++11 attributes can't appear here, despite this context seeming
    // statement-like.
    DiagnoseAndSkipCXX11Attributes();

    if (Tok.isNot(tok::kw_catch))
      return StmtError(Diag(Tok, diag::err_expected_catch));
    while (Tok.is(tok::kw_catch)) {
      StmtResult Handler(ParseCXXCatchBlock(FnTry));
      if (!Handler.isInvalid())
        Handlers.push_back(Handler.get());
    }
    // Don't bother creating the full statement if we don't have any usable
    // handlers.
    if (Handlers.empty())
      return StmtError();

    return Actions.ActOnCXXTryBlock(TryLoc, TryBlock.get(), Handlers);
  }
}

/// ParseCXXCatchBlock - Parse a C++ catch block, called handler in the standard
///
///   handler:
///     'catch' '(' exception-declaration ')' compound-statement
///
///   exception-declaration:
///     attribute-specifier-seq[opt] type-specifier-seq declarator
///     attribute-specifier-seq[opt] type-specifier-seq abstract-declarator[opt]
///     '...'
///
StmtResult Parser::ParseCXXCatchBlock(bool FnCatch) {
  assert(Tok.is(tok::kw_catch) && "Expected 'catch'");

  SourceLocation CatchLoc = ConsumeToken();

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume())
    return StmtError();

  // C++ 3.3.2p3:
  // The name in a catch exception-declaration is local to the handler and
  // shall not be redeclared in the outermost block of the handler.
  ParseScope CatchScope(this, Scope::DeclScope | Scope::ControlScope |
                          (FnCatch ? Scope::FnTryCatchScope : 0));

  // exception-declaration is equivalent to '...' or a parameter-declaration
  // without default arguments.
  Decl *ExceptionDecl = nullptr;
  if (Tok.isNot(tok::ellipsis)) {
    ParsedAttributesWithRange Attributes(AttrFactory);
    MaybeParseCXX11Attributes(Attributes);

    DeclSpec DS(AttrFactory);
    DS.takeAttributesFrom(Attributes);

    if (ParseCXXTypeSpecifierSeq(DS))
      return StmtError();

    Declarator ExDecl(DS, Declarator::CXXCatchContext);
    ParseDeclarator(ExDecl);
    ExceptionDecl = Actions.ActOnExceptionDeclarator(getCurScope(), ExDecl);
  } else
    ConsumeToken();

  T.consumeClose();
  if (T.getCloseLocation().isInvalid())
    return StmtError();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected) << tok::l_brace);

  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  StmtResult Block(ParseCompoundStatement());
  if (Block.isInvalid())
    return Block;

  return Actions.ActOnCXXCatchBlock(CatchLoc, ExceptionDecl, Block.get());
}

void Parser::ParseMicrosoftIfExistsStatement(StmtVector &Stmts) {
  IfExistsCondition Result;
  if (ParseMicrosoftIfExistsCondition(Result))
    return;

  // Handle dependent statements by parsing the braces as a compound statement.
  // This is not the same behavior as Visual C++, which don't treat this as a
  // compound statement, but for Clang's type checking we can't have anything
  // inside these braces escaping to the surrounding code.
  if (Result.Behavior == IEB_Dependent) {
    if (!Tok.is(tok::l_brace)) {
      Diag(Tok, diag::err_expected) << tok::l_brace;
      return;
    }

    StmtResult Compound = ParseCompoundStatement();
    if (Compound.isInvalid())
      return;

    StmtResult DepResult = Actions.ActOnMSDependentExistsStmt(Result.KeywordLoc,
                                                              Result.IsIfExists,
                                                              Result.SS,
                                                              Result.Name,
                                                              Compound.get());
    if (DepResult.isUsable())
      Stmts.push_back(DepResult.get());
    return;
  }

  BalancedDelimiterTracker Braces(*this, tok::l_brace);
  if (Braces.consumeOpen()) {
    Diag(Tok, diag::err_expected) << tok::l_brace;
    return;
  }

  switch (Result.Behavior) {
  case IEB_Parse:
    // Parse the statements below.
    break;

  case IEB_Dependent:
    llvm_unreachable("Dependent case handled above");

  case IEB_Skip:
    Braces.skipToEnd();
    return;
  }

  // Condition is true, parse the statements.
  while (Tok.isNot(tok::r_brace)) {
    StmtResult R = ParseStatementOrDeclaration(Stmts, false);
    if (R.isUsable())
      Stmts.push_back(R.get());
  }
  Braces.consumeClose();
}
