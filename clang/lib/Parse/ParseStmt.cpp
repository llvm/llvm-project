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
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/PrettyDeclStackTrace.h"
#include "clang/Sema/Scope.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// C99 6.8: Statements and Blocks.
//===----------------------------------------------------------------------===//

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
Parser::ParseStatementOrDeclaration(StmtVector &Stmts, bool OnlyStatement) {
  const char *SemiError = 0;
  StmtResult Res;
  
  ParenBraceBracketBalancer BalancerRAIIObj(*this);

  ParsedAttributesWithRange attrs;
  MaybeParseCXX0XAttributes(attrs);

  // Cases in this switch statement should fall through if the parser expects
  // the token to end in a semicolon (in which case SemiError should be set),
  // or they directly 'return;' if not.
  tok::TokenKind Kind  = Tok.getKind();
  SourceLocation AtLoc;
  switch (Kind) {
  case tok::at: // May be a @try or @throw statement
    {
      AtLoc = ConsumeToken();  // consume @
      return ParseObjCAtStatement(AtLoc);
    }

  case tok::code_completion:
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Statement);
    ConsumeCodeCompletionToken();
    return ParseStatementOrDeclaration(Stmts, OnlyStatement);
      
  case tok::identifier:
    if (NextToken().is(tok::colon)) { // C99 6.8.1: labeled-statement
      // identifier ':' statement
      return ParseLabeledStatement(attrs);
    }
    // PASS THROUGH.

  default: {
    if ((getLang().CPlusPlus || !OnlyStatement) && isDeclarationStatement()) {
      SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
      DeclGroupPtrTy Decl = ParseDeclaration(Stmts, Declarator::BlockContext,
                                             DeclEnd, attrs);
      return Actions.ActOnDeclStmt(Decl, DeclStart, DeclEnd);
    }

    if (Tok.is(tok::r_brace)) {
      Diag(Tok, diag::err_expected_statement);
      return StmtError();
    }

    // FIXME: Use the attributes
    // expression[opt] ';'
    ExprResult Expr(ParseExpression());
    if (Expr.isInvalid()) {
      // If the expression is invalid, skip ahead to the next semicolon or '}'.
      // Not doing this opens us up to the possibility of infinite loops if
      // ParseExpression does not consume any tokens.
      SkipUntil(tok::r_brace, /*StopAtSemi=*/true, /*DontConsume=*/true);
      if (Tok.is(tok::semi))
        ConsumeToken();
      return StmtError();
    }
    // Otherwise, eat the semicolon.
    ExpectAndConsumeSemi(diag::err_expected_semi_after_expr);
    return Actions.ActOnExprStmt(Actions.MakeFullExpr(Expr.get()));
  }

  case tok::kw_case:                // C99 6.8.1: labeled-statement
    return ParseCaseStatement(attrs);
  case tok::kw_default:             // C99 6.8.1: labeled-statement
    return ParseDefaultStatement(attrs);

  case tok::l_brace:                // C99 6.8.2: compound-statement
    return ParseCompoundStatement(attrs);
  case tok::semi: {                 // C99 6.8.3p3: expression[opt] ';'
    bool LeadingEmptyMacro = Tok.hasLeadingEmptyMacro();
    return Actions.ActOnNullStmt(ConsumeToken(), LeadingEmptyMacro);
  }

  case tok::kw_if:                  // C99 6.8.4.1: if-statement
    return ParseIfStatement(attrs);
  case tok::kw_switch:              // C99 6.8.4.2: switch-statement
    return ParseSwitchStatement(attrs);

  case tok::kw_while:               // C99 6.8.5.1: while-statement
    return ParseWhileStatement(attrs);
  case tok::kw_do:                  // C99 6.8.5.2: do-statement
    Res = ParseDoStatement(attrs);
    SemiError = "do/while";
    break;
  case tok::kw_for:                 // C99 6.8.5.3: for-statement
    return ParseForStatement(attrs);

  case tok::kw_goto:                // C99 6.8.6.1: goto-statement
    Res = ParseGotoStatement(attrs);
    SemiError = "goto";
    break;
  case tok::kw_continue:            // C99 6.8.6.2: continue-statement
    Res = ParseContinueStatement(attrs);
    SemiError = "continue";
    break;
  case tok::kw_break:               // C99 6.8.6.3: break-statement
    Res = ParseBreakStatement(attrs);
    SemiError = "break";
    break;
  case tok::kw_return:              // C99 6.8.6.4: return-statement
    Res = ParseReturnStatement(attrs);
    SemiError = "return";
    break;

  case tok::kw_asm: {
    ProhibitAttributes(attrs);
    bool msAsm = false;
    Res = ParseAsmStatement(msAsm);
    Res = Actions.ActOnFinishFullStmt(Res.get());
    if (msAsm) return move(Res);
    SemiError = "asm";
    break;
  }

  case tok::kw_try:                 // C++ 15: try-block
    return ParseCXXTryBlock(attrs);
  }

  // If we reached this code, the statement must end in a semicolon.
  if (Tok.is(tok::semi)) {
    ConsumeToken();
  } else if (!Res.isInvalid()) {
    // If the result was valid, then we do want to diagnose this.  Use
    // ExpectAndConsume to emit the diagnostic, even though we know it won't
    // succeed.
    ExpectAndConsume(tok::semi, diag::err_expected_semi_after_stmt, SemiError);
    // Skip until we see a } or ;, but don't eat it.
    SkipUntil(tok::r_brace, true, true);
  }

  return move(Res);
}

/// ParseLabeledStatement - We have an identifier and a ':' after it.
///
///       labeled-statement:
///         identifier ':' statement
/// [GNU]   identifier ':' attributes[opt] statement
///
StmtResult Parser::ParseLabeledStatement(ParsedAttributes &attrs) {
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");

  Token IdentTok = Tok;  // Save the whole token.
  ConsumeToken();  // eat the identifier.

  assert(Tok.is(tok::colon) && "Not a label!");

  // identifier ':' statement
  SourceLocation ColonLoc = ConsumeToken();

  // Read label attributes, if present.
  MaybeParseGNUAttributes(attrs);

  StmtResult SubStmt(ParseStatement());

  // Broken substmt shouldn't prevent the label from being added to the AST.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(ColonLoc);
  
  LabelDecl *LD = Actions.LookupOrCreateLabel(IdentTok.getIdentifierInfo(),
                                              IdentTok.getLocation());
  if (AttributeList *Attrs = attrs.getList())
    Actions.ProcessDeclAttributeList(Actions.CurScope, LD, Attrs);
  
  return Actions.ActOnLabelStmt(IdentTok.getLocation(), LD, ColonLoc,
                                SubStmt.get());
}

/// ParseCaseStatement
///       labeled-statement:
///         'case' constant-expression ':' statement
/// [GNU]   'case' constant-expression '...' constant-expression ':' statement
///
StmtResult Parser::ParseCaseStatement(ParsedAttributes &attrs) {
  assert(Tok.is(tok::kw_case) && "Not a case stmt!");
  // FIXME: Use attributes?

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
  // wierdness in the actions), so this is just local grossness :).

  // TopLevelCase - This is the highest level we have parsed.  'case 1' in the
  // example above.
  StmtResult TopLevelCase(true);

  // DeepestParsedCaseStmt - This is the deepest statement we have parsed, which
  // gets updated each time a new case is parsed, and whose body is unset so
  // far.  When parsing 'case 4', this is the 'case 3' node.
  StmtTy *DeepestParsedCaseStmt = 0;

  // While we have case statements, eat and stack them.
  do {
    SourceLocation CaseLoc = ConsumeToken();  // eat the 'case'.

    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteCase(getCurScope());
      ConsumeCodeCompletionToken();
    }
    
    /// We don't want to treat 'case x : y' as a potential typo for 'case x::y'.
    /// Disable this form of error recovery while we're parsing the case
    /// expression.
    ColonProtectionRAIIObject ColonProtection(*this);
    
    ExprResult LHS(ParseConstantExpression());
    if (LHS.isInvalid()) {
      SkipUntil(tok::colon);
      return StmtError();
    }

    // GNU case range extension.
    SourceLocation DotDotDotLoc;
    ExprResult RHS;
    if (Tok.is(tok::ellipsis)) {
      Diag(Tok, diag::ext_gnu_case_range);
      DotDotDotLoc = ConsumeToken();

      RHS = ParseConstantExpression();
      if (RHS.isInvalid()) {
        SkipUntil(tok::colon);
        return StmtError();
      }
    }
    
    ColonProtection.restore();

    SourceLocation ColonLoc;
    if (Tok.is(tok::colon)) {
      ColonLoc = ConsumeToken();

    // Treat "case blah;" as a typo for "case blah:".
    } else if (Tok.is(tok::semi)) {
      ColonLoc = ConsumeToken();
      Diag(ColonLoc, diag::err_expected_colon_after) << "'case'"
        << FixItHint::CreateReplacement(ColonLoc, ":");
    } else {
      SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
      Diag(ExpectedLoc, diag::err_expected_colon_after) << "'case'"
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
        TopLevelCase = move(Case);
      else
        Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, Case.get());
      DeepestParsedCaseStmt = NextDeepest;
    }

    // Handle all case statements.
  } while (Tok.is(tok::kw_case));

  assert(!TopLevelCase.isInvalid() && "Should have parsed at least one case!");

  // If we found a non-case statement, start by parsing it.
  StmtResult SubStmt;

  if (Tok.isNot(tok::r_brace)) {
    SubStmt = ParseStatement();
  } else {
    // Nicely diagnose the common error "switch (X) { case 4: }", which is
    // not valid.
    // FIXME: add insertion hint.
    Diag(Tok, diag::err_label_end_of_compound_statement);
    SubStmt = true;
  }

  // Broken sub-stmt shouldn't prevent forming the case statement properly.
  if (SubStmt.isInvalid())
    SubStmt = Actions.ActOnNullStmt(SourceLocation());

  // Install the body into the most deeply-nested case.
  Actions.ActOnCaseStmtBody(DeepestParsedCaseStmt, SubStmt.get());

  // Return the top level parsed statement tree.
  return move(TopLevelCase);
}

/// ParseDefaultStatement
///       labeled-statement:
///         'default' ':' statement
/// Note that this does not parse the 'statement' at the end.
///
StmtResult Parser::ParseDefaultStatement(ParsedAttributes &attrs) {
  //FIXME: Use attributes?

  assert(Tok.is(tok::kw_default) && "Not a default stmt!");
  SourceLocation DefaultLoc = ConsumeToken();  // eat the 'default'.

  SourceLocation ColonLoc;
  if (Tok.is(tok::colon)) {
    ColonLoc = ConsumeToken();

  // Treat "default;" as a typo for "default:".
  } else if (Tok.is(tok::semi)) {
    ColonLoc = ConsumeToken();
    Diag(ColonLoc, diag::err_expected_colon_after) << "'default'"
      << FixItHint::CreateReplacement(ColonLoc, ":");
  } else {
    SourceLocation ExpectedLoc = PP.getLocForEndOfToken(PrevTokLocation);
    Diag(ExpectedLoc, diag::err_expected_colon_after) << "'default'"
      << FixItHint::CreateInsertion(ExpectedLoc, ":");
    ColonLoc = ExpectedLoc;
  }
  
  // Diagnose the common error "switch (X) {... default: }", which is not valid.
  if (Tok.is(tok::r_brace)) {
    Diag(Tok, diag::err_label_end_of_compound_statement);
    return StmtError();
  }

  StmtResult SubStmt(ParseStatement());
  if (SubStmt.isInvalid())
    return StmtError();

  return Actions.ActOnDefaultStmt(DefaultLoc, ColonLoc,
                                  SubStmt.get(), getCurScope());
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
/// [OMP]   openmp-directive            [TODO]
///
/// [GNU] label-declarations:
/// [GNU]   label-declaration
/// [GNU]   label-declarations label-declaration
///
/// [GNU] label-declaration:
/// [GNU]   '__label__' identifier-list ';'
///
/// [OMP] openmp-directive:             [TODO]
/// [OMP]   barrier-directive
/// [OMP]   flush-directive
///
StmtResult Parser::ParseCompoundStatement(ParsedAttributes &attrs,
                                                        bool isStmtExpr) {
  //FIXME: Use attributes?

  assert(Tok.is(tok::l_brace) && "Not a compount stmt!");

  // Enter a scope to hold everything within the compound stmt.  Compound
  // statements can always hold declarations.
  ParseScope CompoundScope(this, Scope::DeclScope);

  // Parse the statements in the body.
  return ParseCompoundStatementBody(isStmtExpr);
}


/// ParseCompoundStatementBody - Parse a sequence of statements and invoke the
/// ActOnCompoundStmt action.  This expects the '{' to be the current token, and
/// consume the '}' at the end of the block.  It does not manipulate the scope
/// stack.
StmtResult Parser::ParseCompoundStatementBody(bool isStmtExpr) {
  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(),
                                Tok.getLocation(),
                                "in compound statement ('{}')");
  InMessageExpressionRAIIObject InMessage(*this, false);
  
  SourceLocation LBraceLoc = ConsumeBrace();  // eat the '{'.

  StmtVector Stmts(Actions);

  // "__label__ X, Y, Z;" is the GNU "Local Label" extension.  These are
  // only allowed at the start of a compound stmt regardless of the language.
  while (Tok.is(tok::kw___label__)) {
    SourceLocation LabelLoc = ConsumeToken();
    Diag(LabelLoc, diag::ext_gnu_local_label);
    
    llvm::SmallVector<Decl *, 8> DeclsInGroup;
    while (1) {
      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        break;
      }
      
      IdentifierInfo *II = Tok.getIdentifierInfo();
      SourceLocation IdLoc = ConsumeToken();
      DeclsInGroup.push_back(Actions.LookupOrCreateLabel(II, IdLoc, LabelLoc));
      
      if (!Tok.is(tok::comma))
        break;
      ConsumeToken();
    }
    
    DeclSpec DS;
    DeclGroupPtrTy Res = Actions.FinalizeDeclaratorGroup(getCurScope(), DS,
                                      DeclsInGroup.data(), DeclsInGroup.size());
    StmtResult R = Actions.ActOnDeclStmt(Res, LabelLoc, Tok.getLocation());
    
    ExpectAndConsume(tok::semi, diag::err_expected_semi_declaration);
    if (R.isUsable())
      Stmts.push_back(R.release());
  }
  
  while (Tok.isNot(tok::r_brace) && Tok.isNot(tok::eof)) {
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

      ParsedAttributesWithRange attrs;
      MaybeParseCXX0XAttributes(attrs);

      // If this is the start of a declaration, parse it as such.
      if (isDeclarationStatement()) {
        // __extension__ silences extension warnings in the subdeclaration.
        // FIXME: Save the __extension__ on the decl as a node somehow?
        ExtensionRAIIObject O(Diags);

        SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
        DeclGroupPtrTy Res = ParseDeclaration(Stmts,
                                              Declarator::BlockContext, DeclEnd,
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
        R = Actions.ActOnExprStmt(Actions.MakeFullExpr(Res.get()));
      }
    }

    if (R.isUsable())
      Stmts.push_back(R.release());
  }

  // We broke out of the while loop because we found a '}' or EOF.
  if (Tok.isNot(tok::r_brace)) {
    Diag(Tok, diag::err_expected_rbrace);
    Diag(LBraceLoc, diag::note_matching) << "{";
    return StmtError();
  }

  SourceLocation RBraceLoc = ConsumeBrace();
  return Actions.ActOnCompoundStmt(LBraceLoc, RBraceLoc, move_arg(Stmts),
                                   isStmtExpr);
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
  SourceLocation LParenLoc = ConsumeParen();
  if (getLang().CPlusPlus) 
    ParseCXXCondition(ExprResult, DeclResult, Loc, ConvertToBoolean);
  else {
    ExprResult = ParseExpression();
    DeclResult = 0;
    
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
  MatchRHSPunctuation(tok::r_paren, LParenLoc);
  return false;
}


/// ParseIfStatement
///       if-statement: [C99 6.8.4.1]
///         'if' '(' expression ')' statement
///         'if' '(' expression ')' statement 'else' statement
/// [C++]   'if' '(' condition ')' statement
/// [C++]   'if' '(' condition ')' statement 'else' statement
///
StmtResult Parser::ParseIfStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  assert(Tok.is(tok::kw_if) && "Not an if stmt!");
  SourceLocation IfLoc = ConsumeToken();  // eat the 'if'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "if";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

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
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(CondExp, CondVar, IfLoc, true))
    return StmtError();

  FullExprArg FullCondExp(Actions.MakeFullExpr(CondExp.get()));

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
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the 'then' stmt.
  SourceLocation ThenStmtLoc = Tok.getLocation();
  StmtResult ThenStmt(ParseStatement());

  // Pop the 'if' scope if needed.
  InnerScope.Exit();

  // If it has an else, parse it.
  SourceLocation ElseLoc;
  SourceLocation ElseStmtLoc;
  StmtResult ElseStmt;

  if (Tok.is(tok::kw_else)) {
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
    ParseScope InnerScope(this, Scope::DeclScope,
                          C99orCXX && Tok.isNot(tok::l_brace));

    ElseStmt = ParseStatement();
    
    // Pop the 'else' scope if needed.
    InnerScope.Exit();
  }

  IfScope.Exit();

  // If the condition was invalid, discard the if statement.  We could recover
  // better by replacing it with a valid expr, but don't do that yet.
  if (CondExp.isInvalid() && !CondVar)
    return StmtError();

  // If the then or else stmt is invalid and the other is valid (and present),
  // make turn the invalid one into a null stmt to avoid dropping the other
  // part.  If both are invalid, return error.
  if ((ThenStmt.isInvalid() && ElseStmt.isInvalid()) ||
      (ThenStmt.isInvalid() && ElseStmt.get() == 0) ||
      (ThenStmt.get() == 0  && ElseStmt.isInvalid())) {
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
StmtResult Parser::ParseSwitchStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  assert(Tok.is(tok::kw_switch) && "Not a switch stmt!");
  SourceLocation SwitchLoc = ConsumeToken();  // eat the 'switch'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "switch";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

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
  unsigned ScopeFlags = Scope::BreakScope;
  if (C99orCXX)
    ScopeFlags |= Scope::DeclScope | Scope::ControlScope;
  ParseScope SwitchScope(this, ScopeFlags);

  // Parse the condition.
  ExprResult Cond;
  Decl *CondVar = 0;
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
      SkipUntil(tok::r_brace, false, false);
    } else
      SkipUntil(tok::semi);
    return move(Switch);
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
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

  // Pop the scopes.
  InnerScope.Exit();
  SwitchScope.Exit();

  if (Body.isInvalid())
    // FIXME: Remove the case statement list from the Switch statement.
    Body = Actions.ActOnNullStmt(Tok.getLocation());
  
  return Actions.ActOnFinishSwitchStmt(SwitchLoc, Switch.get(), Body.get());
}

/// ParseWhileStatement
///       while-statement: [C99 6.8.5.1]
///         'while' '(' expression ')' statement
/// [C++]   'while' '(' condition ')' statement
StmtResult Parser::ParseWhileStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  assert(Tok.is(tok::kw_while) && "Not a while stmt!");
  SourceLocation WhileLoc = Tok.getLocation();
  ConsumeToken();  // eat the 'while'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "while";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXX = getLang().C99 || getLang().CPlusPlus;

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
  Decl *CondVar = 0;
  if (ParseParenExprOrCondition(Cond, CondVar, WhileLoc, true))
    return StmtError();

  FullExprArg FullCond(Actions.MakeFullExpr(Cond.get()));

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
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
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXX && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

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
StmtResult Parser::ParseDoStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  assert(Tok.is(tok::kw_do) && "Not a do stmt!");
  SourceLocation DoLoc = ConsumeToken();  // eat the 'do'.

  // C99 6.8.5p5 - In C99, the do statement is a block.  This is not
  // the case for C90.  Start the loop scope.
  unsigned ScopeFlags;
  if (getLang().C99)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope | Scope::DeclScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope DoScope(this, ScopeFlags);

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
  // there is no compound stmt.  C90 does not have this clause. We only do this
  // if the body isn't a compound statement to avoid push/pop in common cases.
  //
  // C++ 6.5p2:
  // The substatement in an iteration-statement implicitly defines a local scope
  // which is entered and exited each time through the loop.
  //
  ParseScope InnerScope(this, Scope::DeclScope,
                        (getLang().C99 || getLang().CPlusPlus) &&
                        Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  if (Tok.isNot(tok::kw_while)) {
    if (!Body.isInvalid()) {
      Diag(Tok, diag::err_expected_while);
      Diag(DoLoc, diag::note_matching) << "do";
      SkipUntil(tok::semi, false, true);
    }
    return StmtError();
  }
  SourceLocation WhileLoc = ConsumeToken();

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "do/while";
    SkipUntil(tok::semi, false, true);
    return StmtError();
  }

  // Parse the parenthesized condition.
  SourceLocation LPLoc = ConsumeParen();
  ExprResult Cond = ParseExpression();
  SourceLocation RPLoc = MatchRHSPunctuation(tok::r_paren, LPLoc);
  DoScope.Exit();

  if (Cond.isInvalid() || Body.isInvalid())
    return StmtError();

  return Actions.ActOnDoStmt(DoLoc, Body.get(), WhileLoc, LPLoc,
                             Cond.get(), RPLoc);
}

/// ParseForStatement
///       for-statement: [C99 6.8.5.3]
///         'for' '(' expr[opt] ';' expr[opt] ';' expr[opt] ')' statement
///         'for' '(' declaration expr[opt] ';' expr[opt] ')' statement
/// [C++]   'for' '(' for-init-statement condition[opt] ';' expression[opt] ')'
/// [C++]       statement
/// [OBJC2] 'for' '(' declaration 'in' expr ')' statement
/// [OBJC2] 'for' '(' expr 'in' expr ')' statement
///
/// [C++] for-init-statement:
/// [C++]   expression-statement
/// [C++]   simple-declaration
///
StmtResult Parser::ParseForStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  assert(Tok.is(tok::kw_for) && "Not a for stmt!");
  SourceLocation ForLoc = ConsumeToken();  // eat the 'for'.

  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "for";
    SkipUntil(tok::semi);
    return StmtError();
  }

  bool C99orCXXorObjC = getLang().C99 || getLang().CPlusPlus || getLang().ObjC1;

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
  unsigned ScopeFlags;
  if (C99orCXXorObjC)
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope |
                 Scope::DeclScope  | Scope::ControlScope;
  else
    ScopeFlags = Scope::BreakScope | Scope::ContinueScope;

  ParseScope ForScope(this, ScopeFlags);

  SourceLocation LParenLoc = ConsumeParen();
  ExprResult Value;

  bool ForEach = false;
  StmtResult FirstPart;
  bool SecondPartIsInvalid = false;
  FullExprArg SecondPart(Actions);
  ExprResult Collection;
  FullExprArg ThirdPart(Actions);
  Decl *SecondVar = 0;
  
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(), 
                                     C99orCXXorObjC? Sema::PCC_ForInit
                                                   : Sema::PCC_Expression);
    ConsumeCodeCompletionToken();
  }
  
  // Parse the first part of the for specifier.
  if (Tok.is(tok::semi)) {  // for (;
    // no first part, eat the ';'.
    ConsumeToken();
  } else if (isSimpleDeclaration()) {  // for (int X = 4;
    // Parse declaration, which eats the ';'.
    if (!C99orCXXorObjC)   // Use of C99-style for loops in C90 mode?
      Diag(Tok, diag::ext_c99_variable_decl_in_for_loop);

    ParsedAttributesWithRange attrs;
    MaybeParseCXX0XAttributes(attrs);

    SourceLocation DeclStart = Tok.getLocation(), DeclEnd;
    StmtVector Stmts(Actions);
    DeclGroupPtrTy DG = ParseSimpleDeclaration(Stmts, Declarator::ForContext, 
                                               DeclEnd, attrs, false);
    FirstPart = Actions.ActOnDeclStmt(DG, DeclStart, Tok.getLocation());

    if (Tok.is(tok::semi)) {  // for (int x = 4;
      ConsumeToken();
    } else if ((ForEach = isTokIdentifier_in())) {
      Actions.ActOnForEachDeclStmt(DG);
      // ObjC: for (id x in expr)
      ConsumeToken(); // consume 'in'
      
      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DG);
        ConsumeCodeCompletionToken();
      }
      Collection = ParseExpression();
    } else {
      Diag(Tok, diag::err_expected_semi_for);
    }
  } else {
    Value = ParseExpression();

    ForEach = isTokIdentifier_in();

    // Turn the expression into a stmt.
    if (!Value.isInvalid()) {
      if (ForEach)
        FirstPart = Actions.ActOnForEachLValueExpr(Value.get());
      else
        FirstPart = Actions.ActOnExprStmt(Actions.MakeFullExpr(Value.get()));
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    } else if (ForEach) {
      ConsumeToken(); // consume 'in'
      
      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteObjCForCollection(getCurScope(), DeclGroupPtrTy());
        ConsumeCodeCompletionToken();
      }
      Collection = ParseExpression();
    } else {
      if (!Value.isInvalid()) {
        Diag(Tok, diag::err_expected_semi_for);
      } else {
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, true, true);
        if (Tok.is(tok::semi))
          ConsumeToken();
      }
    }
  }
  if (!ForEach) {
    assert(!SecondPart.get() && "Shouldn't have a second expression yet.");
    // Parse the second part of the for specifier.
    if (Tok.is(tok::semi)) {  // for (...;;
      // no second part.
    } else if (Tok.is(tok::r_paren)) {
      // missing both semicolons.
    } else {
      ExprResult Second;
      if (getLang().CPlusPlus)
        ParseCXXCondition(Second, SecondVar, ForLoc, true);
      else {
        Second = ParseExpression();
        if (!Second.isInvalid())
          Second = Actions.ActOnBooleanCondition(getCurScope(), ForLoc, 
                                                 Second.get());
      }
      SecondPartIsInvalid = Second.isInvalid();
      SecondPart = Actions.MakeFullExpr(Second.get());
    }

    if (Tok.isNot(tok::semi)) {
      if (!SecondPartIsInvalid || SecondVar)
        Diag(Tok, diag::err_expected_semi_for);
      else
        // Skip until semicolon or rparen, don't consume it.
        SkipUntil(tok::r_paren, true, true);
    }

    if (Tok.is(tok::semi)) {
      ConsumeToken();
    }

    // Parse the third part of the for specifier.
    if (Tok.isNot(tok::r_paren)) {   // for (...;...;)
      ExprResult Third = ParseExpression();
      ThirdPart = Actions.MakeFullExpr(Third.take());
    }
  }
  // Match the ')'.
  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, LParenLoc);

  // C99 6.8.5p5 - In C99, the body of the if statement is a scope, even if
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
  ParseScope InnerScope(this, Scope::DeclScope,
                        C99orCXXorObjC && Tok.isNot(tok::l_brace));

  // Read the body statement.
  StmtResult Body(ParseStatement());

  // Pop the body scope if needed.
  InnerScope.Exit();

  // Leave the for-scope.
  ForScope.Exit();

  if (Body.isInvalid())
    return StmtError();

  if (!ForEach)
    return Actions.ActOnForStmt(ForLoc, LParenLoc, FirstPart.take(), SecondPart,
                                SecondVar, ThirdPart, RParenLoc, Body.take());

  // FIXME: It isn't clear how to communicate the late destruction of 
  // C++ temporaries used to create the collection.
  return Actions.ActOnObjCForCollectionStmt(ForLoc, LParenLoc, FirstPart.take(), 
                                            Collection.take(), RParenLoc, 
                                            Body.take());
}

/// ParseGotoStatement
///       jump-statement:
///         'goto' identifier ';'
/// [GNU]   'goto' '*' expression ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseGotoStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

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
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
    Res = Actions.ActOnIndirectGotoStmt(GotoLoc, StarLoc, R.take());
  } else {
    Diag(Tok, diag::err_expected_ident);
    return StmtError();
  }

  return move(Res);
}

/// ParseContinueStatement
///       jump-statement:
///         'continue' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseContinueStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  SourceLocation ContinueLoc = ConsumeToken();  // eat the 'continue'.
  return Actions.ActOnContinueStmt(ContinueLoc, getCurScope());
}

/// ParseBreakStatement
///       jump-statement:
///         'break' ';'
///
/// Note: this lets the caller parse the end ';'.
///
StmtResult Parser::ParseBreakStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  SourceLocation BreakLoc = ConsumeToken();  // eat the 'break'.
  return Actions.ActOnBreakStmt(BreakLoc, getCurScope());
}

/// ParseReturnStatement
///       jump-statement:
///         'return' expression[opt] ';'
StmtResult Parser::ParseReturnStatement(ParsedAttributes &attrs) {
  // FIXME: Use attributes?

  assert(Tok.is(tok::kw_return) && "Not a return stmt!");
  SourceLocation ReturnLoc = ConsumeToken();  // eat the 'return'.

  ExprResult R;
  if (Tok.isNot(tok::semi)) {
    if (Tok.is(tok::code_completion)) {
      Actions.CodeCompleteReturn(getCurScope());
      ConsumeCodeCompletionToken();
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
        
    // FIXME: This is a hack to allow something like C++0x's generalized
    // initializer lists, but only enough of this feature to allow Clang to
    // parse libstdc++ 4.5's headers.
    if (Tok.is(tok::l_brace) && getLang().CPlusPlus) {
      R = ParseInitializer();
      if (R.isUsable() && !getLang().CPlusPlus0x)
        Diag(R.get()->getLocStart(), diag::ext_generalized_initializer_lists)
          << R.get()->getSourceRange();
    } else
        R = ParseExpression();
    if (R.isInvalid()) {  // Skip to the semicolon, but don't consume it.
      SkipUntil(tok::semi, false, true);
      return StmtError();
    }
  }
  return Actions.ActOnReturnStmt(ReturnLoc, R.take());
}

/// FuzzyParseMicrosoftAsmStatement. When -fms-extensions is enabled, this
/// routine is called to skip/ignore tokens that comprise the MS asm statement.
StmtResult Parser::FuzzyParseMicrosoftAsmStatement(SourceLocation AsmLoc) {
  SourceLocation EndLoc;
  if (Tok.is(tok::l_brace)) {
    unsigned short savedBraceCount = BraceCount;
    do {
      EndLoc = Tok.getLocation();
      ConsumeAnyToken();
    } while (BraceCount > savedBraceCount && Tok.isNot(tok::eof));
  } else {
    // From the MS website: If used without braces, the __asm keyword means
    // that the rest of the line is an assembly-language statement.
    SourceManager &SrcMgr = PP.getSourceManager();
    SourceLocation TokLoc = Tok.getLocation();
    unsigned LineNo = SrcMgr.getInstantiationLineNumber(TokLoc);
    do {
      EndLoc = TokLoc;
      ConsumeAnyToken();
      TokLoc = Tok.getLocation();
    } while ((SrcMgr.getInstantiationLineNumber(TokLoc) == LineNo) &&
             Tok.isNot(tok::r_brace) && Tok.isNot(tok::semi) &&
             Tok.isNot(tok::eof));
  }
  Token t;
  t.setKind(tok::string_literal);
  t.setLiteralData("\"/*FIXME: not done*/\"");
  t.clearFlag(Token::NeedsCleaning);
  t.setLength(21);
  ExprResult AsmString(Actions.ActOnStringLiteral(&t, 1));
  ExprVector Constraints(Actions);
  ExprVector Exprs(Actions);
  ExprVector Clobbers(Actions);
  return Actions.ActOnAsmStmt(AsmLoc, true, true, 0, 0, 0,
                              move_arg(Constraints), move_arg(Exprs),
                              AsmString.take(), move_arg(Clobbers),
                              EndLoc, true);
}

/// ParseAsmStatement - Parse a GNU extended asm statement.
///       asm-statement:
///         gnu-asm-statement
///         ms-asm-statement
///
/// [GNU] gnu-asm-statement:
///         'asm' type-qualifier[opt] '(' asm-argument ')' ';'
///
/// [GNU] asm-argument:
///         asm-string-literal
///         asm-string-literal ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///         asm-string-literal ':' asm-operands[opt] ':' asm-operands[opt]
///                 ':' asm-clobbers
///
/// [GNU] asm-clobbers:
///         asm-string-literal
///         asm-clobbers ',' asm-string-literal
///
/// [MS]  ms-asm-statement:
///         '__asm' assembly-instruction ';'[opt]
///         '__asm' '{' assembly-instruction-list '}' ';'[opt]
///
/// [MS]  assembly-instruction-list:
///         assembly-instruction ';'[opt]
///         assembly-instruction-list ';' assembly-instruction ';'[opt]
///
StmtResult Parser::ParseAsmStatement(bool &msAsm) {
  assert(Tok.is(tok::kw_asm) && "Not an asm stmt");
  SourceLocation AsmLoc = ConsumeToken();

  if (getLang().Microsoft && Tok.isNot(tok::l_paren) && !isTypeQualifier()) {
    msAsm = true;
    return FuzzyParseMicrosoftAsmStatement(AsmLoc);
  }
  DeclSpec DS;
  SourceLocation Loc = Tok.getLocation();
  ParseTypeQualifierListOpt(DS, true, false);

  // GNU asms accept, but warn, about type-qualifiers other than volatile.
  if (DS.getTypeQualifiers() & DeclSpec::TQ_const)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "const";
  if (DS.getTypeQualifiers() & DeclSpec::TQ_restrict)
    Diag(Loc, diag::w_asm_qualifier_ignored) << "restrict";

  // Remember if this was a volatile asm.
  bool isVolatile = DS.getTypeQualifiers() & DeclSpec::TQ_volatile;
  if (Tok.isNot(tok::l_paren)) {
    Diag(Tok, diag::err_expected_lparen_after) << "asm";
    SkipUntil(tok::r_paren);
    return StmtError();
  }
  Loc = ConsumeParen();

  ExprResult AsmString(ParseAsmStringLiteral());
  if (AsmString.isInvalid())
    return StmtError();

  llvm::SmallVector<IdentifierInfo *, 4> Names;
  ExprVector Constraints(Actions);
  ExprVector Exprs(Actions);
  ExprVector Clobbers(Actions);

  if (Tok.is(tok::r_paren)) {
    // We have a simple asm expression like 'asm("foo")'.
    SourceLocation RParenLoc = ConsumeParen();
    return Actions.ActOnAsmStmt(AsmLoc, /*isSimple*/ true, isVolatile,
                                /*NumOutputs*/ 0, /*NumInputs*/ 0, 0, 
                                move_arg(Constraints), move_arg(Exprs),
                                AsmString.take(), move_arg(Clobbers),
                                RParenLoc);
  }

  // Parse Outputs, if present.
  bool AteExtraColon = false;
  if (Tok.is(tok::colon) || Tok.is(tok::coloncolon)) {
    // In C++ mode, parse "::" like ": :".
    AteExtraColon = Tok.is(tok::coloncolon);
    ConsumeToken();
  
    if (!AteExtraColon &&
        ParseAsmOperandsOpt(Names, Constraints, Exprs))
      return StmtError();
  }
  
  unsigned NumOutputs = Names.size();

  // Parse Inputs, if present.
  if (AteExtraColon ||
      Tok.is(tok::colon) || Tok.is(tok::coloncolon)) {
    // In C++ mode, parse "::" like ": :".
    if (AteExtraColon)
      AteExtraColon = false;
    else {
      AteExtraColon = Tok.is(tok::coloncolon);
      ConsumeToken();
    }
    
    if (!AteExtraColon &&
        ParseAsmOperandsOpt(Names, Constraints, Exprs))
      return StmtError();
  }

  assert(Names.size() == Constraints.size() &&
         Constraints.size() == Exprs.size() &&
         "Input operand size mismatch!");

  unsigned NumInputs = Names.size() - NumOutputs;

  // Parse the clobbers, if present.
  if (AteExtraColon || Tok.is(tok::colon)) {
    if (!AteExtraColon)
      ConsumeToken();

    // Parse the asm-string list for clobbers if present.
    if (Tok.isNot(tok::r_paren)) {
      while (1) {
        ExprResult Clobber(ParseAsmStringLiteral());

        if (Clobber.isInvalid())
          break;

        Clobbers.push_back(Clobber.release());

        if (Tok.isNot(tok::comma)) break;
        ConsumeToken();
      }
    }
  }

  SourceLocation RParenLoc = MatchRHSPunctuation(tok::r_paren, Loc);
  return Actions.ActOnAsmStmt(AsmLoc, false, isVolatile,
                              NumOutputs, NumInputs, Names.data(),
                              move_arg(Constraints), move_arg(Exprs),
                              AsmString.take(), move_arg(Clobbers),
                              RParenLoc);
}

/// ParseAsmOperands - Parse the asm-operands production as used by
/// asm-statement, assuming the leading ':' token was eaten.
///
/// [GNU] asm-operands:
///         asm-operand
///         asm-operands ',' asm-operand
///
/// [GNU] asm-operand:
///         asm-string-literal '(' expression ')'
///         '[' identifier ']' asm-string-literal '(' expression ')'
///
//
// FIXME: Avoid unnecessary std::string trashing.
bool Parser::ParseAsmOperandsOpt(llvm::SmallVectorImpl<IdentifierInfo *> &Names,
                                 llvm::SmallVectorImpl<ExprTy *> &Constraints,
                                 llvm::SmallVectorImpl<ExprTy *> &Exprs) {
  // 'asm-operands' isn't present?
  if (!isTokenStringLiteral() && Tok.isNot(tok::l_square))
    return false;

  while (1) {
    // Read the [id] if present.
    if (Tok.is(tok::l_square)) {
      SourceLocation Loc = ConsumeBracket();

      if (Tok.isNot(tok::identifier)) {
        Diag(Tok, diag::err_expected_ident);
        SkipUntil(tok::r_paren);
        return true;
      }

      IdentifierInfo *II = Tok.getIdentifierInfo();
      ConsumeToken();

      Names.push_back(II);
      MatchRHSPunctuation(tok::r_square, Loc);
    } else
      Names.push_back(0);

    ExprResult Constraint(ParseAsmStringLiteral());
    if (Constraint.isInvalid()) {
        SkipUntil(tok::r_paren);
        return true;
    }
    Constraints.push_back(Constraint.release());

    if (Tok.isNot(tok::l_paren)) {
      Diag(Tok, diag::err_expected_lparen_after) << "asm operand";
      SkipUntil(tok::r_paren);
      return true;
    }

    // Read the parenthesized expression.
    SourceLocation OpenLoc = ConsumeParen();
    ExprResult Res(ParseExpression());
    MatchRHSPunctuation(tok::r_paren, OpenLoc);
    if (Res.isInvalid()) {
      SkipUntil(tok::r_paren);
      return true;
    }
    Exprs.push_back(Res.release());
    // Eat the comma and continue parsing if it exists.
    if (Tok.isNot(tok::comma)) return false;
    ConsumeToken();
  }

  return true;
}

Decl *Parser::ParseFunctionStatementBody(Decl *Decl, ParseScope &BodyScope) {
  assert(Tok.is(tok::l_brace));
  SourceLocation LBraceLoc = Tok.getLocation();

  if (PP.isCodeCompletionEnabled()) {
    if (trySkippingFunctionBodyForCodeCompletion()) {
      BodyScope.Exit();
      return Actions.ActOnFinishFunctionBody(Decl, 0);
    }
  }
  
  PrettyDeclStackTraceEntry CrashInfo(Actions, Decl, LBraceLoc,
                                      "parsing function body");

  // Do not enter a scope for the brace, as the arguments are in the same scope
  // (the function body) as the body itself.  Instead, just read the statement
  // list and put it into a CompoundStmt for safe keeping.
  StmtResult FnBody(ParseCompoundStatementBody());

  // If the function body could not be parsed, make a bogus compoundstmt.
  if (FnBody.isInvalid())
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc,
                                       MultiStmtArg(Actions), false);

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.take());
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

  if (PP.isCodeCompletionEnabled()) {
    if (trySkippingFunctionBodyForCodeCompletion()) {
      BodyScope.Exit();
      return Actions.ActOnFinishFunctionBody(Decl, 0);
    }
  }

  SourceLocation LBraceLoc = Tok.getLocation();
  StmtResult FnBody(ParseCXXTryBlockCommon(TryLoc));
  // If we failed to parse the try-catch, we just give the function an empty
  // compound statement as the body.
  if (FnBody.isInvalid())
    FnBody = Actions.ActOnCompoundStmt(LBraceLoc, LBraceLoc,
                                       MultiStmtArg(Actions), false);

  BodyScope.Exit();
  return Actions.ActOnFinishFunctionBody(Decl, FnBody.take());
}

bool Parser::trySkippingFunctionBodyForCodeCompletion() {
  assert(Tok.is(tok::l_brace));
  assert(PP.isCodeCompletionEnabled() &&
         "Should only be called when in code-completion mode");

  // We're in code-completion mode. Skip parsing for all function bodies unless
  // the body contains the code-completion point.
  TentativeParsingAction PA(*this);
  ConsumeBrace();
  if (SkipUntil(tok::r_brace, /*StopAtSemi=*/false, /*DontConsume=*/false,
                /*StopAtCodeCompletion=*/true)) {
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
StmtResult Parser::ParseCXXTryBlock(ParsedAttributes &attrs) {
  // FIXME: Add attributes?

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
StmtResult Parser::ParseCXXTryBlockCommon(SourceLocation TryLoc) {
  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));
  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  ParsedAttributesWithRange attrs;
  StmtResult TryBlock(ParseCompoundStatement(attrs));
  if (TryBlock.isInvalid())
    return move(TryBlock);

  StmtVector Handlers(Actions);
  MaybeParseCXX0XAttributes(attrs);
  ProhibitAttributes(attrs);

  if (Tok.isNot(tok::kw_catch))
    return StmtError(Diag(Tok, diag::err_expected_catch));
  while (Tok.is(tok::kw_catch)) {
    StmtResult Handler(ParseCXXCatchBlock());
    if (!Handler.isInvalid())
      Handlers.push_back(Handler.release());
  }
  // Don't bother creating the full statement if we don't have any usable
  // handlers.
  if (Handlers.empty())
    return StmtError();

  return Actions.ActOnCXXTryBlock(TryLoc, TryBlock.take(), move_arg(Handlers));
}

/// ParseCXXCatchBlock - Parse a C++ catch block, called handler in the standard
///
///       handler:
///         'catch' '(' exception-declaration ')' compound-statement
///
///       exception-declaration:
///         type-specifier-seq declarator
///         type-specifier-seq abstract-declarator
///         type-specifier-seq
///         '...'
///
StmtResult Parser::ParseCXXCatchBlock() {
  assert(Tok.is(tok::kw_catch) && "Expected 'catch'");

  SourceLocation CatchLoc = ConsumeToken();

  SourceLocation LParenLoc = Tok.getLocation();
  if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen))
    return StmtError();

  // C++ 3.3.2p3:
  // The name in a catch exception-declaration is local to the handler and
  // shall not be redeclared in the outermost block of the handler.
  ParseScope CatchScope(this, Scope::DeclScope | Scope::ControlScope);

  // exception-declaration is equivalent to '...' or a parameter-declaration
  // without default arguments.
  Decl *ExceptionDecl = 0;
  if (Tok.isNot(tok::ellipsis)) {
    DeclSpec DS;
    if (ParseCXXTypeSpecifierSeq(DS))
      return StmtError();
    Declarator ExDecl(DS, Declarator::CXXCatchContext);
    ParseDeclarator(ExDecl);
    ExceptionDecl = Actions.ActOnExceptionDeclarator(getCurScope(), ExDecl);
  } else
    ConsumeToken();

  if (MatchRHSPunctuation(tok::r_paren, LParenLoc).isInvalid())
    return StmtError();

  if (Tok.isNot(tok::l_brace))
    return StmtError(Diag(Tok, diag::err_expected_lbrace));

  // FIXME: Possible draft standard bug: attribute-specifier should be allowed?
  ParsedAttributes attrs;
  StmtResult Block(ParseCompoundStatement(attrs));
  if (Block.isInvalid())
    return move(Block);

  return Actions.ActOnCXXCatchBlock(CatchLoc, ExceptionDecl, Block.take());
}
