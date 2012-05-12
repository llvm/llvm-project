//===--- ParseExpr.cpp - Expression Parsing -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Expression parsing implementation.  Expressions in
// C99 basically consist of a bunch of binary operators with unary operators and
// other random stuff at the leaves.
//
// In the C99 grammar, these unary operators bind tightest and are represented
// as the 'cast-expression' production.  Everything else is either a binary
// operator (e.g. '/') or a ternary operator ("?:").  The unary leaves are
// handled by ParseCastExpression, the higher level pieces are handled by
// ParseBinaryExpression.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/TypoCorrection.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "RAIIObjectsForParser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallString.h"
using namespace clang;

/// getBinOpPrecedence - Return the precedence of the specified binary operator
/// token.
static prec::Level getBinOpPrecedence(tok::TokenKind Kind,
                                      bool GreaterThanIsOperator,
                                      bool CPlusPlus0x) {
  switch (Kind) {
  case tok::greater:
    // C++ [temp.names]p3:
    //   [...] When parsing a template-argument-list, the first
    //   non-nested > is taken as the ending delimiter rather than a
    //   greater-than operator. [...]
    if (GreaterThanIsOperator)
      return prec::Relational;
    return prec::Unknown;

  case tok::greatergreater:
    // C++0x [temp.names]p3:
    //
    //   [...] Similarly, the first non-nested >> is treated as two
    //   consecutive but distinct > tokens, the first of which is
    //   taken as the end of the template-argument-list and completes
    //   the template-id. [...]
    if (GreaterThanIsOperator || !CPlusPlus0x)
      return prec::Shift;
    return prec::Unknown;

  default:                        return prec::Unknown;
  case tok::comma:                return prec::Comma;
  case tok::equal:
  case tok::starequal:
  case tok::slashequal:
  case tok::percentequal:
  case tok::plusequal:
  case tok::minusequal:
  case tok::lesslessequal:
  case tok::greatergreaterequal:
  case tok::ampequal:
  case tok::caretequal:
  case tok::pipeequal:            return prec::Assignment;
  case tok::question:             return prec::Conditional;
  case tok::pipepipe:             return prec::LogicalOr;
  case tok::ampamp:               return prec::LogicalAnd;
  case tok::pipe:                 return prec::InclusiveOr;
  case tok::caret:                return prec::ExclusiveOr;
  case tok::amp:                  return prec::And;
  case tok::exclaimequal:
  case tok::equalequal:           return prec::Equality;
  case tok::lessequal:
  case tok::less:
  case tok::greaterequal:         return prec::Relational;
  case tok::lessless:             return prec::Shift;
  case tok::plus:
  case tok::minus:                return prec::Additive;
  case tok::percent:
  case tok::slash:
  case tok::star:                 return prec::Multiplicative;
  case tok::periodstar:
  case tok::arrowstar:            return prec::PointerToMember;
  }
}


/// ParseExpression - Simple precedence-based parser for binary/ternary
/// operators.
///
/// Note: we diverge from the C99 grammar when parsing the assignment-expression
/// production.  C99 specifies that the LHS of an assignment operator should be
/// parsed as a unary-expression, but consistency dictates that it be a
/// conditional-expession.  In practice, the important thing here is that the
/// LHS of an assignment has to be an l-value, which productions between
/// unary-expression and conditional-expression don't produce.  Because we want
/// consistency, we parse the LHS as a conditional-expression, then check for
/// l-value-ness in semantic analysis stages.
///
///       pm-expression: [C++ 5.5]
///         cast-expression
///         pm-expression '.*' cast-expression
///         pm-expression '->*' cast-expression
///
///       multiplicative-expression: [C99 6.5.5]
///     Note: in C++, apply pm-expression instead of cast-expression
///         cast-expression
///         multiplicative-expression '*' cast-expression
///         multiplicative-expression '/' cast-expression
///         multiplicative-expression '%' cast-expression
///
///       additive-expression: [C99 6.5.6]
///         multiplicative-expression
///         additive-expression '+' multiplicative-expression
///         additive-expression '-' multiplicative-expression
///
///       shift-expression: [C99 6.5.7]
///         additive-expression
///         shift-expression '<<' additive-expression
///         shift-expression '>>' additive-expression
///
///       relational-expression: [C99 6.5.8]
///         shift-expression
///         relational-expression '<' shift-expression
///         relational-expression '>' shift-expression
///         relational-expression '<=' shift-expression
///         relational-expression '>=' shift-expression
///
///       equality-expression: [C99 6.5.9]
///         relational-expression
///         equality-expression '==' relational-expression
///         equality-expression '!=' relational-expression
///
///       AND-expression: [C99 6.5.10]
///         equality-expression
///         AND-expression '&' equality-expression
///
///       exclusive-OR-expression: [C99 6.5.11]
///         AND-expression
///         exclusive-OR-expression '^' AND-expression
///
///       inclusive-OR-expression: [C99 6.5.12]
///         exclusive-OR-expression
///         inclusive-OR-expression '|' exclusive-OR-expression
///
///       logical-AND-expression: [C99 6.5.13]
///         inclusive-OR-expression
///         logical-AND-expression '&&' inclusive-OR-expression
///
///       logical-OR-expression: [C99 6.5.14]
///         logical-AND-expression
///         logical-OR-expression '||' logical-AND-expression
///
///       conditional-expression: [C99 6.5.15]
///         logical-OR-expression
///         logical-OR-expression '?' expression ':' conditional-expression
/// [GNU]   logical-OR-expression '?' ':' conditional-expression
/// [C++] the third operand is an assignment-expression
///
///       assignment-expression: [C99 6.5.16]
///         conditional-expression
///         unary-expression assignment-operator assignment-expression
/// [C++]   throw-expression [C++ 15]
///
///       assignment-operator: one of
///         = *= /= %= += -= <<= >>= &= ^= |=
///
///       expression: [C99 6.5.17]
///         assignment-expression ...[opt]
///         expression ',' assignment-expression ...[opt]
ExprResult Parser::ParseExpression(TypeCastState isTypeCast) {
  ExprResult LHS(ParseAssignmentExpression(isTypeCast));
  return ParseRHSOfBinaryExpression(move(LHS), prec::Comma);
}

/// This routine is called when the '@' is seen and consumed.
/// Current token is an Identifier and is not a 'try'. This
/// routine is necessary to disambiguate @try-statement from,
/// for example, @encode-expression.
///
ExprResult
Parser::ParseExpressionWithLeadingAt(SourceLocation AtLoc) {
  ExprResult LHS(ParseObjCAtExpression(AtLoc));
  return ParseRHSOfBinaryExpression(move(LHS), prec::Comma);
}

/// This routine is called when a leading '__extension__' is seen and
/// consumed.  This is necessary because the token gets consumed in the
/// process of disambiguating between an expression and a declaration.
ExprResult
Parser::ParseExpressionWithLeadingExtension(SourceLocation ExtLoc) {
  ExprResult LHS(true);
  {
    // Silence extension warnings in the sub-expression
    ExtensionRAIIObject O(Diags);

    LHS = ParseCastExpression(false);
  }

  if (!LHS.isInvalid())
    LHS = Actions.ActOnUnaryOp(getCurScope(), ExtLoc, tok::kw___extension__,
                               LHS.take());

  return ParseRHSOfBinaryExpression(move(LHS), prec::Comma);
}

/// ParseAssignmentExpression - Parse an expr that doesn't include commas.
ExprResult Parser::ParseAssignmentExpression(TypeCastState isTypeCast) {
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Expression);
    cutOffParsing();
    return ExprError();
  }

  if (Tok.is(tok::kw_throw))
    return ParseThrowExpression();

  ExprResult LHS = ParseCastExpression(/*isUnaryExpression=*/false,
                                       /*isAddressOfOperand=*/false,
                                       isTypeCast);
  return ParseRHSOfBinaryExpression(move(LHS), prec::Assignment);
}

/// ParseAssignmentExprWithObjCMessageExprStart - Parse an assignment expression
/// where part of an objc message send has already been parsed.  In this case
/// LBracLoc indicates the location of the '[' of the message send, and either
/// ReceiverName or ReceiverExpr is non-null indicating the receiver of the
/// message.
///
/// Since this handles full assignment-expression's, it handles postfix
/// expressions and other binary operators for these expressions as well.
ExprResult
Parser::ParseAssignmentExprWithObjCMessageExprStart(SourceLocation LBracLoc,
                                                    SourceLocation SuperLoc,
                                                    ParsedType ReceiverType,
                                                    Expr *ReceiverExpr) {
  ExprResult R
    = ParseObjCMessageExpressionBody(LBracLoc, SuperLoc,
                                     ReceiverType, ReceiverExpr);
  R = ParsePostfixExpressionSuffix(R);
  return ParseRHSOfBinaryExpression(R, prec::Assignment);
}


ExprResult Parser::ParseConstantExpression(TypeCastState isTypeCast) {
  // C++03 [basic.def.odr]p2:
  //   An expression is potentially evaluated unless it appears where an
  //   integral constant expression is required (see 5.19) [...].
  // C++98 and C++11 have no such rule, but this is only a defect in C++98.
  EnterExpressionEvaluationContext Unevaluated(Actions,
                                               Sema::ConstantEvaluated);

  ExprResult LHS(ParseCastExpression(false, false, isTypeCast));
  ExprResult Res(ParseRHSOfBinaryExpression(LHS, prec::Conditional));
  return Actions.ActOnConstantExpression(Res);
}

/// ParseRHSOfBinaryExpression - Parse a binary expression that starts with
/// LHS and has a precedence of at least MinPrec.
ExprResult
Parser::ParseRHSOfBinaryExpression(ExprResult LHS, prec::Level MinPrec) {
  prec::Level NextTokPrec = getBinOpPrecedence(Tok.getKind(),
                                               GreaterThanIsOperator,
                                               getLangOpts().CPlusPlus0x);
  SourceLocation ColonLoc;

  while (1) {
    // If this token has a lower precedence than we are allowed to parse (e.g.
    // because we are called recursively, or because the token is not a binop),
    // then we are done!
    if (NextTokPrec < MinPrec)
      return move(LHS);

    // Consume the operator, saving the operator token for error reporting.
    Token OpToken = Tok;
    ConsumeToken();

    // Special case handling for the ternary operator.
    ExprResult TernaryMiddle(true);
    if (NextTokPrec == prec::Conditional) {
      if (Tok.isNot(tok::colon)) {
        // Don't parse FOO:BAR as if it were a typo for FOO::BAR.
        ColonProtectionRAIIObject X(*this);

        // Handle this production specially:
        //   logical-OR-expression '?' expression ':' conditional-expression
        // In particular, the RHS of the '?' is 'expression', not
        // 'logical-OR-expression' as we might expect.
        TernaryMiddle = ParseExpression();
        if (TernaryMiddle.isInvalid()) {
          LHS = ExprError();
          TernaryMiddle = 0;
        }
      } else {
        // Special case handling of "X ? Y : Z" where Y is empty:
        //   logical-OR-expression '?' ':' conditional-expression   [GNU]
        TernaryMiddle = 0;
        Diag(Tok, diag::ext_gnu_conditional_expr);
      }

      if (Tok.is(tok::colon)) {
        // Eat the colon.
        ColonLoc = ConsumeToken();
      } else {
        // Otherwise, we're missing a ':'.  Assume that this was a typo that
        // the user forgot. If we're not in a macro expansion, we can suggest
        // a fixit hint. If there were two spaces before the current token,
        // suggest inserting the colon in between them, otherwise insert ": ".
        SourceLocation FILoc = Tok.getLocation();
        const char *FIText = ": ";
        const SourceManager &SM = PP.getSourceManager();
        if (FILoc.isFileID() || PP.isAtStartOfMacroExpansion(FILoc, &FILoc)) {
          assert(FILoc.isFileID());
          bool IsInvalid = false;
          const char *SourcePtr =
            SM.getCharacterData(FILoc.getLocWithOffset(-1), &IsInvalid);
          if (!IsInvalid && *SourcePtr == ' ') {
            SourcePtr =
              SM.getCharacterData(FILoc.getLocWithOffset(-2), &IsInvalid);
            if (!IsInvalid && *SourcePtr == ' ') {
              FILoc = FILoc.getLocWithOffset(-1);
              FIText = ":";
            }
          }
        }
        
        Diag(Tok, diag::err_expected_colon)
          << FixItHint::CreateInsertion(FILoc, FIText);
        Diag(OpToken, diag::note_matching) << "?";
        ColonLoc = Tok.getLocation();
      }
    }
    
    // Code completion for the right-hand side of an assignment expression
    // goes through a special hook that takes the left-hand side into account.
    if (Tok.is(tok::code_completion) && NextTokPrec == prec::Assignment) {
      Actions.CodeCompleteAssignmentRHS(getCurScope(), LHS.get());
      cutOffParsing();
      return ExprError();
    }
    
    // Parse another leaf here for the RHS of the operator.
    // ParseCastExpression works here because all RHS expressions in C have it
    // as a prefix, at least. However, in C++, an assignment-expression could
    // be a throw-expression, which is not a valid cast-expression.
    // Therefore we need some special-casing here.
    // Also note that the third operand of the conditional operator is
    // an assignment-expression in C++, and in C++11, we can have a
    // braced-init-list on the RHS of an assignment. For better diagnostics,
    // parse as if we were allowed braced-init-lists everywhere, and check that
    // they only appear on the RHS of assignments later.
    ExprResult RHS;
    bool RHSIsInitList = false;
    if (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace)) {
      RHS = ParseBraceInitializer();
      RHSIsInitList = true;
    } else if (getLangOpts().CPlusPlus && NextTokPrec <= prec::Conditional)
      RHS = ParseAssignmentExpression();
    else
      RHS = ParseCastExpression(false);

    if (RHS.isInvalid())
      LHS = ExprError();
    
    // Remember the precedence of this operator and get the precedence of the
    // operator immediately to the right of the RHS.
    prec::Level ThisPrec = NextTokPrec;
    NextTokPrec = getBinOpPrecedence(Tok.getKind(), GreaterThanIsOperator,
                                     getLangOpts().CPlusPlus0x);

    // Assignment and conditional expressions are right-associative.
    bool isRightAssoc = ThisPrec == prec::Conditional ||
                        ThisPrec == prec::Assignment;

    // Get the precedence of the operator to the right of the RHS.  If it binds
    // more tightly with RHS than we do, evaluate it completely first.
    if (ThisPrec < NextTokPrec ||
        (ThisPrec == NextTokPrec && isRightAssoc)) {
      if (!RHS.isInvalid() && RHSIsInitList) {
        Diag(Tok, diag::err_init_list_bin_op)
          << /*LHS*/0 << PP.getSpelling(Tok) << Actions.getExprRange(RHS.get());
        RHS = ExprError();
      }
      // If this is left-associative, only parse things on the RHS that bind
      // more tightly than the current operator.  If it is left-associative, it
      // is okay, to bind exactly as tightly.  For example, compile A=B=C=D as
      // A=(B=(C=D)), where each paren is a level of recursion here.
      // The function takes ownership of the RHS.
      RHS = ParseRHSOfBinaryExpression(RHS, 
                            static_cast<prec::Level>(ThisPrec + !isRightAssoc));
      RHSIsInitList = false;

      if (RHS.isInvalid())
        LHS = ExprError();

      NextTokPrec = getBinOpPrecedence(Tok.getKind(), GreaterThanIsOperator,
                                       getLangOpts().CPlusPlus0x);
    }
    assert(NextTokPrec <= ThisPrec && "Recursion didn't work!");

    if (!RHS.isInvalid() && RHSIsInitList) {
      if (ThisPrec == prec::Assignment) {
        Diag(OpToken, diag::warn_cxx98_compat_generalized_initializer_lists)
          << Actions.getExprRange(RHS.get());
      } else {
        Diag(OpToken, diag::err_init_list_bin_op)
          << /*RHS*/1 << PP.getSpelling(OpToken)
          << Actions.getExprRange(RHS.get());
        LHS = ExprError();
      }
    }

    if (!LHS.isInvalid()) {
      // Combine the LHS and RHS into the LHS (e.g. build AST).
      if (TernaryMiddle.isInvalid()) {
        // If we're using '>>' as an operator within a template
        // argument list (in C++98), suggest the addition of
        // parentheses so that the code remains well-formed in C++0x.
        if (!GreaterThanIsOperator && OpToken.is(tok::greatergreater))
          SuggestParentheses(OpToken.getLocation(),
                             diag::warn_cxx0x_right_shift_in_template_arg,
                         SourceRange(Actions.getExprRange(LHS.get()).getBegin(),
                                     Actions.getExprRange(RHS.get()).getEnd()));

        LHS = Actions.ActOnBinOp(getCurScope(), OpToken.getLocation(),
                                 OpToken.getKind(), LHS.take(), RHS.take());
      } else
        LHS = Actions.ActOnConditionalOp(OpToken.getLocation(), ColonLoc,
                                         LHS.take(), TernaryMiddle.take(),
                                         RHS.take());
    }
  }
}

/// ParseCastExpression - Parse a cast-expression, or, if isUnaryExpression is
/// true, parse a unary-expression. isAddressOfOperand exists because an
/// id-expression that is the operand of address-of gets special treatment
/// due to member pointers.
///
ExprResult Parser::ParseCastExpression(bool isUnaryExpression,
                                       bool isAddressOfOperand,
                                       TypeCastState isTypeCast) {
  bool NotCastExpr;
  ExprResult Res = ParseCastExpression(isUnaryExpression,
                                       isAddressOfOperand,
                                       NotCastExpr,
                                       isTypeCast);
  if (NotCastExpr)
    Diag(Tok, diag::err_expected_expression);
  return move(Res);
}

namespace {
class CastExpressionIdValidator : public CorrectionCandidateCallback {
 public:
  CastExpressionIdValidator(bool AllowTypes, bool AllowNonTypes)
      : AllowNonTypes(AllowNonTypes) {
    WantTypeSpecifiers = AllowTypes;
  }

  virtual bool ValidateCandidate(const TypoCorrection &candidate) {
    NamedDecl *ND = candidate.getCorrectionDecl();
    if (!ND)
      return candidate.isKeyword();

    if (isa<TypeDecl>(ND))
      return WantTypeSpecifiers;
    return AllowNonTypes;
  }

 private:
  bool AllowNonTypes;
};
}

/// ParseCastExpression - Parse a cast-expression, or, if isUnaryExpression is
/// true, parse a unary-expression. isAddressOfOperand exists because an
/// id-expression that is the operand of address-of gets special treatment
/// due to member pointers. NotCastExpr is set to true if the token is not the
/// start of a cast-expression, and no diagnostic is emitted in this case.
///
///       cast-expression: [C99 6.5.4]
///         unary-expression
///         '(' type-name ')' cast-expression
///
///       unary-expression:  [C99 6.5.3]
///         postfix-expression
///         '++' unary-expression
///         '--' unary-expression
///         unary-operator cast-expression
///         'sizeof' unary-expression
///         'sizeof' '(' type-name ')'
/// [C++11] 'sizeof' '...' '(' identifier ')'
/// [GNU]   '__alignof' unary-expression
/// [GNU]   '__alignof' '(' type-name ')'
/// [C++11] 'alignof' '(' type-id ')'
/// [GNU]   '&&' identifier
/// [C++11] 'noexcept' '(' expression ')' [C++11 5.3.7]
/// [C++]   new-expression
/// [C++]   delete-expression
///
///       unary-operator: one of
///         '&'  '*'  '+'  '-'  '~'  '!'
/// [GNU]   '__extension__'  '__real'  '__imag'
///
///       primary-expression: [C99 6.5.1]
/// [C99]   identifier
/// [C++]   id-expression
///         constant
///         string-literal
/// [C++]   boolean-literal  [C++ 2.13.5]
/// [C++11] 'nullptr'        [C++11 2.14.7]
/// [C++11] user-defined-literal
///         '(' expression ')'
/// [C11]   generic-selection
///         '__func__'        [C99 6.4.2.2]
/// [GNU]   '__FUNCTION__'
/// [GNU]   '__PRETTY_FUNCTION__'
/// [GNU]   '(' compound-statement ')'
/// [GNU]   '__builtin_va_arg' '(' assignment-expression ',' type-name ')'
/// [GNU]   '__builtin_offsetof' '(' type-name ',' offsetof-member-designator')'
/// [GNU]   '__builtin_choose_expr' '(' assign-expr ',' assign-expr ','
///                                     assign-expr ')'
/// [GNU]   '__builtin_types_compatible_p' '(' type-name ',' type-name ')'
/// [GNU]   '__null'
/// [OBJC]  '[' objc-message-expr ']'
/// [OBJC]  '@selector' '(' objc-selector-arg ')'
/// [OBJC]  '@protocol' '(' identifier ')'
/// [OBJC]  '@encode' '(' type-name ')'
/// [OBJC]  objc-string-literal
/// [C++]   simple-type-specifier '(' expression-list[opt] ')'      [C++ 5.2.3]
/// [C++11] simple-type-specifier braced-init-list                  [C++11 5.2.3]
/// [C++]   typename-specifier '(' expression-list[opt] ')'         [C++ 5.2.3]
/// [C++11] typename-specifier braced-init-list                     [C++11 5.2.3]
/// [C++]   'const_cast' '<' type-name '>' '(' expression ')'       [C++ 5.2p1]
/// [C++]   'dynamic_cast' '<' type-name '>' '(' expression ')'     [C++ 5.2p1]
/// [C++]   'reinterpret_cast' '<' type-name '>' '(' expression ')' [C++ 5.2p1]
/// [C++]   'static_cast' '<' type-name '>' '(' expression ')'      [C++ 5.2p1]
/// [C++]   'typeid' '(' expression ')'                             [C++ 5.2p1]
/// [C++]   'typeid' '(' type-id ')'                                [C++ 5.2p1]
/// [C++]   'this'          [C++ 9.3.2]
/// [G++]   unary-type-trait '(' type-id ')'
/// [G++]   binary-type-trait '(' type-id ',' type-id ')'           [TODO]
/// [EMBT]  array-type-trait '(' type-id ',' integer ')'
/// [clang] '^' block-literal
///
///       constant: [C99 6.4.4]
///         integer-constant
///         floating-constant
///         enumeration-constant -> identifier
///         character-constant
///
///       id-expression: [C++ 5.1]
///                   unqualified-id
///                   qualified-id          
///
///       unqualified-id: [C++ 5.1]
///                   identifier
///                   operator-function-id
///                   conversion-function-id
///                   '~' class-name        
///                   template-id           
///
///       new-expression: [C++ 5.3.4]
///                   '::'[opt] 'new' new-placement[opt] new-type-id
///                                     new-initializer[opt]
///                   '::'[opt] 'new' new-placement[opt] '(' type-id ')'
///                                     new-initializer[opt]
///
///       delete-expression: [C++ 5.3.5]
///                   '::'[opt] 'delete' cast-expression
///                   '::'[opt] 'delete' '[' ']' cast-expression
///
/// [GNU/Embarcadero] unary-type-trait:
///                   '__is_arithmetic'
///                   '__is_floating_point'
///                   '__is_integral'
///                   '__is_lvalue_expr'
///                   '__is_rvalue_expr'
///                   '__is_complete_type'
///                   '__is_void'
///                   '__is_array'
///                   '__is_function'
///                   '__is_reference'
///                   '__is_lvalue_reference'
///                   '__is_rvalue_reference'
///                   '__is_fundamental'
///                   '__is_object'
///                   '__is_scalar'
///                   '__is_compound'
///                   '__is_pointer'
///                   '__is_member_object_pointer'
///                   '__is_member_function_pointer'
///                   '__is_member_pointer'
///                   '__is_const'
///                   '__is_volatile'
///                   '__is_trivial'
///                   '__is_standard_layout'
///                   '__is_signed'
///                   '__is_unsigned'
///
/// [GNU] unary-type-trait:
///                   '__has_nothrow_assign'
///                   '__has_nothrow_copy'
///                   '__has_nothrow_constructor'
///                   '__has_trivial_assign'                  [TODO]
///                   '__has_trivial_copy'                    [TODO]
///                   '__has_trivial_constructor'
///                   '__has_trivial_destructor'
///                   '__has_virtual_destructor'
///                   '__is_abstract'                         [TODO]
///                   '__is_class'
///                   '__is_empty'                            [TODO]
///                   '__is_enum'
///                   '__is_final'
///                   '__is_pod'
///                   '__is_polymorphic'
///                   '__is_trivial'
///                   '__is_union'
///
/// [Clang] unary-type-trait:
///                   '__trivially_copyable'
///
///       binary-type-trait:
/// [GNU]             '__is_base_of'       
/// [MS]              '__is_convertible_to'
///                   '__is_convertible'
///                   '__is_same'
///
/// [Embarcadero] array-type-trait:
///                   '__array_rank'
///                   '__array_extent'
///
/// [Embarcadero] expression-trait:
///                   '__is_lvalue_expr'
///                   '__is_rvalue_expr'
///
ExprResult Parser::ParseCastExpression(bool isUnaryExpression,
                                       bool isAddressOfOperand,
                                       bool &NotCastExpr,
                                       TypeCastState isTypeCast) {
  ExprResult Res;
  tok::TokenKind SavedKind = Tok.getKind();
  NotCastExpr = false;

  // This handles all of cast-expression, unary-expression, postfix-expression,
  // and primary-expression.  We handle them together like this for efficiency
  // and to simplify handling of an expression starting with a '(' token: which
  // may be one of a parenthesized expression, cast-expression, compound literal
  // expression, or statement expression.
  //
  // If the parsed tokens consist of a primary-expression, the cases below
  // break out of the switch;  at the end we call ParsePostfixExpressionSuffix
  // to handle the postfix expression suffixes.  Cases that cannot be followed
  // by postfix exprs should return without invoking
  // ParsePostfixExpressionSuffix.
  switch (SavedKind) {
  case tok::l_paren: {
    // If this expression is limited to being a unary-expression, the parent can
    // not start a cast expression.
    ParenParseOption ParenExprType =
      (isUnaryExpression && !getLangOpts().CPlusPlus)? CompoundLiteral : CastExpr;
    ParsedType CastTy;
    SourceLocation RParenLoc;
    
    {
      // The inside of the parens don't need to be a colon protected scope, and
      // isn't immediately a message send.
      ColonProtectionRAIIObject X(*this, false);

      Res = ParseParenExpression(ParenExprType, false/*stopIfCastExr*/,
                                 isTypeCast == IsTypeCast, CastTy, RParenLoc);
    }

    switch (ParenExprType) {
    case SimpleExpr:   break;    // Nothing else to do.
    case CompoundStmt: break;  // Nothing else to do.
    case CompoundLiteral:
      // We parsed '(' type-name ')' '{' ... '}'.  If any suffixes of
      // postfix-expression exist, parse them now.
      break;
    case CastExpr:
      // We have parsed the cast-expression and no postfix-expr pieces are
      // following.
      return move(Res);
    }

    break;
  }

    // primary-expression
  case tok::numeric_constant:
    // constant: integer-constant
    // constant: floating-constant

    Res = Actions.ActOnNumericConstant(Tok, /*UDLScope*/getCurScope());
    ConsumeToken();
    break;

  case tok::kw_true:
  case tok::kw_false:
    return ParseCXXBoolLiteral();
  
  case tok::kw___objc_yes:
  case tok::kw___objc_no:
      return ParseObjCBoolLiteral();

  case tok::kw_nullptr:
    Diag(Tok, diag::warn_cxx98_compat_nullptr);
    return Actions.ActOnCXXNullPtrLiteral(ConsumeToken());

  case tok::annot_primary_expr:
    assert(Res.get() == 0 && "Stray primary-expression annotation?");
    Res = getExprAnnotation(Tok);
    ConsumeToken();
    break;
      
  case tok::kw_decltype:
  case tok::identifier: {      // primary-expression: identifier
                               // unqualified-id: identifier
                               // constant: enumeration-constant
    // Turn a potentially qualified name into a annot_typename or
    // annot_cxxscope if it would be valid.  This handles things like x::y, etc.
    if (getLangOpts().CPlusPlus) {
      // Avoid the unnecessary parse-time lookup in the common case
      // where the syntax forbids a type.
      const Token &Next = NextToken();
      if (Next.is(tok::coloncolon) ||
          (!ColonIsSacred && Next.is(tok::colon)) ||
          Next.is(tok::less) ||
          Next.is(tok::l_paren) ||
          Next.is(tok::l_brace)) {
        // If TryAnnotateTypeOrScopeToken annotates the token, tail recurse.
        if (TryAnnotateTypeOrScopeToken())
          return ExprError();
        if (!Tok.is(tok::identifier))
          return ParseCastExpression(isUnaryExpression, isAddressOfOperand);
      }
    }

    // Consume the identifier so that we can see if it is followed by a '(' or
    // '.'.
    IdentifierInfo &II = *Tok.getIdentifierInfo();
    SourceLocation ILoc = ConsumeToken();
    
    // Support 'Class.property' and 'super.property' notation.
    if (getLangOpts().ObjC1 && Tok.is(tok::period) &&
        (Actions.getTypeName(II, ILoc, getCurScope()) ||
         // Allow the base to be 'super' if in an objc-method.
         (&II == Ident_super && getCurScope()->isInObjcMethodScope()))) {
      ConsumeToken();
      
      // Allow either an identifier or the keyword 'class' (in C++).
      if (Tok.isNot(tok::identifier) && 
          !(getLangOpts().CPlusPlus && Tok.is(tok::kw_class))) {
        Diag(Tok, diag::err_expected_property_name);
        return ExprError();
      }
      IdentifierInfo &PropertyName = *Tok.getIdentifierInfo();
      SourceLocation PropertyLoc = ConsumeToken();
      
      Res = Actions.ActOnClassPropertyRefExpr(II, PropertyName,
                                              ILoc, PropertyLoc);
      break;
    }

    // In an Objective-C method, if we have "super" followed by an identifier,
    // the token sequence is ill-formed. However, if there's a ':' or ']' after
    // that identifier, this is probably a message send with a missing open
    // bracket. Treat it as such. 
    if (getLangOpts().ObjC1 && &II == Ident_super && !InMessageExpression &&
        getCurScope()->isInObjcMethodScope() &&
        ((Tok.is(tok::identifier) &&
         (NextToken().is(tok::colon) || NextToken().is(tok::r_square))) ||
         Tok.is(tok::code_completion))) {
      Res = ParseObjCMessageExpressionBody(SourceLocation(), ILoc, ParsedType(), 
                                           0);
      break;
    }
    
    // If we have an Objective-C class name followed by an identifier
    // and either ':' or ']', this is an Objective-C class message
    // send that's missing the opening '['. Recovery
    // appropriately. Also take this path if we're performing code
    // completion after an Objective-C class name.
    if (getLangOpts().ObjC1 && 
        ((Tok.is(tok::identifier) && !InMessageExpression) || 
         Tok.is(tok::code_completion))) {
      const Token& Next = NextToken();
      if (Tok.is(tok::code_completion) || 
          Next.is(tok::colon) || Next.is(tok::r_square))
        if (ParsedType Typ = Actions.getTypeName(II, ILoc, getCurScope()))
          if (Typ.get()->isObjCObjectOrInterfaceType()) {
            // Fake up a Declarator to use with ActOnTypeName.
            DeclSpec DS(AttrFactory);
            DS.SetRangeStart(ILoc);
            DS.SetRangeEnd(ILoc);
            const char *PrevSpec = 0;
            unsigned DiagID;
            DS.SetTypeSpecType(TST_typename, ILoc, PrevSpec, DiagID, Typ);
            
            Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
            TypeResult Ty = Actions.ActOnTypeName(getCurScope(), 
                                                  DeclaratorInfo);
            if (Ty.isInvalid())
              break;
            
            Res = ParseObjCMessageExpressionBody(SourceLocation(), 
                                                 SourceLocation(), 
                                                 Ty.get(), 0);
            break;
          }
    }
    
    // Make sure to pass down the right value for isAddressOfOperand.
    if (isAddressOfOperand && isPostfixExpressionSuffixStart())
      isAddressOfOperand = false;
   
    // Function designators are allowed to be undeclared (C99 6.5.1p2), so we
    // need to know whether or not this identifier is a function designator or
    // not.
    UnqualifiedId Name;
    CXXScopeSpec ScopeSpec;
    SourceLocation TemplateKWLoc;
    CastExpressionIdValidator Validator(isTypeCast != NotTypeCast,
                                        isTypeCast != IsTypeCast);
    Name.setIdentifier(&II, ILoc);
    Res = Actions.ActOnIdExpression(getCurScope(), ScopeSpec, TemplateKWLoc,
                                    Name, Tok.is(tok::l_paren),
                                    isAddressOfOperand, &Validator);
    break;
  }
  case tok::char_constant:     // constant: character-constant
  case tok::wide_char_constant:
  case tok::utf16_char_constant:
  case tok::utf32_char_constant:
    Res = Actions.ActOnCharacterConstant(Tok, /*UDLScope*/getCurScope());
    ConsumeToken();
    break;
  case tok::kw___func__:       // primary-expression: __func__ [C99 6.4.2.2]
  case tok::kw___FUNCTION__:   // primary-expression: __FUNCTION__ [GNU]
  case tok::kw___PRETTY_FUNCTION__:  // primary-expression: __P..Y_F..N__ [GNU]
    Res = Actions.ActOnPredefinedExpr(Tok.getLocation(), SavedKind);
    ConsumeToken();
    break;
  case tok::string_literal:    // primary-expression: string-literal
  case tok::wide_string_literal:
  case tok::utf8_string_literal:
  case tok::utf16_string_literal:
  case tok::utf32_string_literal:
    Res = ParseStringLiteralExpression(true);
    break;
  case tok::kw__Generic:   // primary-expression: generic-selection [C11 6.5.1]
    Res = ParseGenericSelectionExpression();
    break;
  case tok::kw___builtin_va_arg:
  case tok::kw___builtin_offsetof:
  case tok::kw___builtin_choose_expr:
  case tok::kw___builtin_astype: // primary-expression: [OCL] as_type()
    return ParseBuiltinPrimaryExpression();
  case tok::kw___null:
    return Actions.ActOnGNUNullExpr(ConsumeToken());

  case tok::plusplus:      // unary-expression: '++' unary-expression [C99]
  case tok::minusminus: {  // unary-expression: '--' unary-expression [C99]
    // C++ [expr.unary] has:
    //   unary-expression:
    //     ++ cast-expression
    //     -- cast-expression
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(!getLangOpts().CPlusPlus);
    if (!Res.isInvalid())
      Res = Actions.ActOnUnaryOp(getCurScope(), SavedLoc, SavedKind, Res.get());
    return move(Res);
  }
  case tok::amp: {         // unary-expression: '&' cast-expression
    // Special treatment because of member pointers
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(false, true);
    if (!Res.isInvalid())
      Res = Actions.ActOnUnaryOp(getCurScope(), SavedLoc, SavedKind, Res.get());
    return move(Res);
  }

  case tok::star:          // unary-expression: '*' cast-expression
  case tok::plus:          // unary-expression: '+' cast-expression
  case tok::minus:         // unary-expression: '-' cast-expression
  case tok::tilde:         // unary-expression: '~' cast-expression
  case tok::exclaim:       // unary-expression: '!' cast-expression
  case tok::kw___real:     // unary-expression: '__real' cast-expression [GNU]
  case tok::kw___imag: {   // unary-expression: '__imag' cast-expression [GNU]
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(false);
    if (!Res.isInvalid())
      Res = Actions.ActOnUnaryOp(getCurScope(), SavedLoc, SavedKind, Res.get());
    return move(Res);
  }

  case tok::kw___extension__:{//unary-expression:'__extension__' cast-expr [GNU]
    // __extension__ silences extension warnings in the subexpression.
    ExtensionRAIIObject O(Diags);  // Use RAII to do this.
    SourceLocation SavedLoc = ConsumeToken();
    Res = ParseCastExpression(false);
    if (!Res.isInvalid())
      Res = Actions.ActOnUnaryOp(getCurScope(), SavedLoc, SavedKind, Res.get());
    return move(Res);
  }
  case tok::kw_sizeof:     // unary-expression: 'sizeof' unary-expression
                           // unary-expression: 'sizeof' '(' type-name ')'
  case tok::kw_alignof:
  case tok::kw___alignof:  // unary-expression: '__alignof' unary-expression
                           // unary-expression: '__alignof' '(' type-name ')'
                           // unary-expression: 'alignof' '(' type-id ')'
  case tok::kw_vec_step:   // unary-expression: OpenCL 'vec_step' expression
    return ParseUnaryExprOrTypeTraitExpression();
  case tok::ampamp: {      // unary-expression: '&&' identifier
    SourceLocation AmpAmpLoc = ConsumeToken();
    if (Tok.isNot(tok::identifier))
      return ExprError(Diag(Tok, diag::err_expected_ident));

    if (getCurScope()->getFnParent() == 0)
      return ExprError(Diag(Tok, diag::err_address_of_label_outside_fn));
    
    Diag(AmpAmpLoc, diag::ext_gnu_address_of_label);
    LabelDecl *LD = Actions.LookupOrCreateLabel(Tok.getIdentifierInfo(),
                                                Tok.getLocation());
    Res = Actions.ActOnAddrLabel(AmpAmpLoc, Tok.getLocation(), LD);
    ConsumeToken();
    return move(Res);
  }
  case tok::kw_const_cast:
  case tok::kw_dynamic_cast:
  case tok::kw_reinterpret_cast:
  case tok::kw_static_cast:
    Res = ParseCXXCasts();
    break;
  case tok::kw_typeid:
    Res = ParseCXXTypeid();
    break;
  case tok::kw___uuidof:
    Res = ParseCXXUuidof();
    break;
  case tok::kw_this:
    Res = ParseCXXThis();
    break;

  case tok::annot_typename:
    if (isStartOfObjCClassMessageMissingOpenBracket()) {
      ParsedType Type = getTypeAnnotation(Tok);
      
      // Fake up a Declarator to use with ActOnTypeName.
      DeclSpec DS(AttrFactory);
      DS.SetRangeStart(Tok.getLocation());
      DS.SetRangeEnd(Tok.getLastLoc());

      const char *PrevSpec = 0;
      unsigned DiagID;
      DS.SetTypeSpecType(TST_typename, Tok.getAnnotationEndLoc(),
                         PrevSpec, DiagID, Type);
      
      Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
      TypeResult Ty = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
      if (Ty.isInvalid())
        break;

      ConsumeToken();
      Res = ParseObjCMessageExpressionBody(SourceLocation(), SourceLocation(),
                                           Ty.get(), 0);
      break;
    }
    // Fall through
      
  case tok::annot_decltype:
  case tok::kw_char:
  case tok::kw_wchar_t:
  case tok::kw_char16_t:
  case tok::kw_char32_t:
  case tok::kw_bool:
  case tok::kw_short:
  case tok::kw_int:
  case tok::kw_long:
  case tok::kw___int64:
  case tok::kw___int128:
  case tok::kw_signed:
  case tok::kw_unsigned:
  case tok::kw_half:
  case tok::kw_float:
  case tok::kw_double:
  case tok::kw_void:
  case tok::kw_typename:
  case tok::kw_typeof:
  case tok::kw___vector: {
    if (!getLangOpts().CPlusPlus) {
      Diag(Tok, diag::err_expected_expression);
      return ExprError();
    }

    if (SavedKind == tok::kw_typename) {
      // postfix-expression: typename-specifier '(' expression-list[opt] ')'
      //                     typename-specifier braced-init-list
      if (TryAnnotateTypeOrScopeToken())
        return ExprError();
    }

    // postfix-expression: simple-type-specifier '(' expression-list[opt] ')'
    //                     simple-type-specifier braced-init-list
    //
    DeclSpec DS(AttrFactory);
    ParseCXXSimpleTypeSpecifier(DS);
    if (Tok.isNot(tok::l_paren) &&
        (!getLangOpts().CPlusPlus0x || Tok.isNot(tok::l_brace)))
      return ExprError(Diag(Tok, diag::err_expected_lparen_after_type)
                         << DS.getSourceRange());

    if (Tok.is(tok::l_brace))
      Diag(Tok, diag::warn_cxx98_compat_generalized_initializer_lists);

    Res = ParseCXXTypeConstructExpression(DS);
    break;
  }

  case tok::annot_cxxscope: { // [C++] id-expression: qualified-id
    // If TryAnnotateTypeOrScopeToken annotates the token, tail recurse.
    // (We can end up in this situation after tentative parsing.)
    if (TryAnnotateTypeOrScopeToken())
      return ExprError();
    if (!Tok.is(tok::annot_cxxscope))
      return ParseCastExpression(isUnaryExpression, isAddressOfOperand,
                                 NotCastExpr, isTypeCast);

    Token Next = NextToken();
    if (Next.is(tok::annot_template_id)) {
      TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Next);
      if (TemplateId->Kind == TNK_Type_template) {
        // We have a qualified template-id that we know refers to a
        // type, translate it into a type and continue parsing as a
        // cast expression.
        CXXScopeSpec SS;
        ParseOptionalCXXScopeSpecifier(SS, ParsedType(), 
                                       /*EnteringContext=*/false);
        AnnotateTemplateIdTokenAsType();
        return ParseCastExpression(isUnaryExpression, isAddressOfOperand,
                                   NotCastExpr, isTypeCast);
      }
    }

    // Parse as an id-expression.
    Res = ParseCXXIdExpression(isAddressOfOperand);
    break;
  }

  case tok::annot_template_id: { // [C++]          template-id
    TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Tok);
    if (TemplateId->Kind == TNK_Type_template) {
      // We have a template-id that we know refers to a type,
      // translate it into a type and continue parsing as a cast
      // expression.
      AnnotateTemplateIdTokenAsType();
      return ParseCastExpression(isUnaryExpression, isAddressOfOperand,
                                 NotCastExpr, isTypeCast);
    }

    // Fall through to treat the template-id as an id-expression.
  }

  case tok::kw_operator: // [C++] id-expression: operator/conversion-function-id
    Res = ParseCXXIdExpression(isAddressOfOperand);
    break;

  case tok::coloncolon: {
    // ::foo::bar -> global qualified name etc.   If TryAnnotateTypeOrScopeToken
    // annotates the token, tail recurse.
    if (TryAnnotateTypeOrScopeToken())
      return ExprError();
    if (!Tok.is(tok::coloncolon))
      return ParseCastExpression(isUnaryExpression, isAddressOfOperand);

    // ::new -> [C++] new-expression
    // ::delete -> [C++] delete-expression
    SourceLocation CCLoc = ConsumeToken();
    if (Tok.is(tok::kw_new))
      return ParseCXXNewExpression(true, CCLoc);
    if (Tok.is(tok::kw_delete))
      return ParseCXXDeleteExpression(true, CCLoc);

    // This is not a type name or scope specifier, it is an invalid expression.
    Diag(CCLoc, diag::err_expected_expression);
    return ExprError();
  }

  case tok::kw_new: // [C++] new-expression
    return ParseCXXNewExpression(false, Tok.getLocation());

  case tok::kw_delete: // [C++] delete-expression
    return ParseCXXDeleteExpression(false, Tok.getLocation());

  case tok::kw_noexcept: { // [C++0x] 'noexcept' '(' expression ')'
    Diag(Tok, diag::warn_cxx98_compat_noexcept_expr);
    SourceLocation KeyLoc = ConsumeToken();
    BalancedDelimiterTracker T(*this, tok::l_paren);

    if (T.expectAndConsume(diag::err_expected_lparen_after, "noexcept"))
      return ExprError();
    // C++11 [expr.unary.noexcept]p1:
    //   The noexcept operator determines whether the evaluation of its operand,
    //   which is an unevaluated operand, can throw an exception.
    EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated);
    ExprResult Result = ParseExpression();

    T.consumeClose();

    if (!Result.isInvalid())
      Result = Actions.ActOnNoexceptExpr(KeyLoc, T.getOpenLocation(), 
                                         Result.take(), T.getCloseLocation());
    return move(Result);
  }

  case tok::kw___is_abstract: // [GNU] unary-type-trait
  case tok::kw___is_class:
  case tok::kw___is_empty:
  case tok::kw___is_enum:
  case tok::kw___is_literal:
  case tok::kw___is_arithmetic:
  case tok::kw___is_integral:
  case tok::kw___is_floating_point:
  case tok::kw___is_complete_type:
  case tok::kw___is_void:
  case tok::kw___is_array:
  case tok::kw___is_function:
  case tok::kw___is_reference:
  case tok::kw___is_lvalue_reference:
  case tok::kw___is_rvalue_reference:
  case tok::kw___is_fundamental:
  case tok::kw___is_object:
  case tok::kw___is_scalar:
  case tok::kw___is_compound:
  case tok::kw___is_pointer:
  case tok::kw___is_member_object_pointer:
  case tok::kw___is_member_function_pointer:
  case tok::kw___is_member_pointer:
  case tok::kw___is_const:
  case tok::kw___is_volatile:
  case tok::kw___is_standard_layout:
  case tok::kw___is_signed:
  case tok::kw___is_unsigned:
  case tok::kw___is_literal_type:
  case tok::kw___is_pod:
  case tok::kw___is_polymorphic:
  case tok::kw___is_trivial:
  case tok::kw___is_trivially_copyable:
  case tok::kw___is_union:
  case tok::kw___is_final:
  case tok::kw___has_trivial_constructor:
  case tok::kw___has_trivial_copy:
  case tok::kw___has_trivial_assign:
  case tok::kw___has_trivial_destructor:
  case tok::kw___has_nothrow_assign:
  case tok::kw___has_nothrow_copy:
  case tok::kw___has_nothrow_constructor:
  case tok::kw___has_virtual_destructor:
    return ParseUnaryTypeTrait();

  case tok::kw___builtin_types_compatible_p:
  case tok::kw___is_base_of:
  case tok::kw___is_same:
  case tok::kw___is_convertible:
  case tok::kw___is_convertible_to:
  case tok::kw___is_trivially_assignable:
    return ParseBinaryTypeTrait();

  case tok::kw___is_trivially_constructible:
    return ParseTypeTrait();
      
  case tok::kw___array_rank:
  case tok::kw___array_extent:
    return ParseArrayTypeTrait();

  case tok::kw___is_lvalue_expr:
  case tok::kw___is_rvalue_expr:
    return ParseExpressionTrait();
      
  case tok::at: {
    SourceLocation AtLoc = ConsumeToken();
    return ParseObjCAtExpression(AtLoc);
  }
  case tok::caret:
    Res = ParseBlockLiteralExpression();
    break;
  case tok::code_completion: {
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Expression);
    cutOffParsing();
    return ExprError();
  }
  case tok::l_square:
    if (getLangOpts().CPlusPlus0x) {
      if (getLangOpts().ObjC1) {
        // C++11 lambda expressions and Objective-C message sends both start with a
        // square bracket.  There are three possibilities here:
        // we have a valid lambda expression, we have an invalid lambda
        // expression, or we have something that doesn't appear to be a lambda.
        // If we're in the last case, we fall back to ParseObjCMessageExpression.
        Res = TryParseLambdaExpression();
        if (!Res.isInvalid() && !Res.get())
          Res = ParseObjCMessageExpression();
        break;
      }
      Res = ParseLambdaExpression();
      break;
    }
    if (getLangOpts().ObjC1) {
      Res = ParseObjCMessageExpression();
      break;
    }
    // FALL THROUGH.
  default:
    NotCastExpr = true;
    return ExprError();
  }

  // These can be followed by postfix-expr pieces.
  return ParsePostfixExpressionSuffix(Res);
}

/// ParsePostfixExpressionSuffix - Once the leading part of a postfix-expression
/// is parsed, this method parses any suffixes that apply.
///
///       postfix-expression: [C99 6.5.2]
///         primary-expression
///         postfix-expression '[' expression ']'
///         postfix-expression '[' braced-init-list ']'
///         postfix-expression '(' argument-expression-list[opt] ')'
///         postfix-expression '.' identifier
///         postfix-expression '->' identifier
///         postfix-expression '++'
///         postfix-expression '--'
///         '(' type-name ')' '{' initializer-list '}'
///         '(' type-name ')' '{' initializer-list ',' '}'
///
///       argument-expression-list: [C99 6.5.2]
///         argument-expression ...[opt]
///         argument-expression-list ',' assignment-expression ...[opt]
///
ExprResult
Parser::ParsePostfixExpressionSuffix(ExprResult LHS) {
  // Now that the primary-expression piece of the postfix-expression has been
  // parsed, see if there are any postfix-expression pieces here.
  SourceLocation Loc;
  while (1) {
    switch (Tok.getKind()) {
    case tok::code_completion:
      if (InMessageExpression)
        return move(LHS);
        
      Actions.CodeCompletePostfixExpression(getCurScope(), LHS);
      cutOffParsing();
      return ExprError();
        
    case tok::identifier:
      // If we see identifier: after an expression, and we're not already in a
      // message send, then this is probably a message send with a missing
      // opening bracket '['.
      if (getLangOpts().ObjC1 && !InMessageExpression && 
          (NextToken().is(tok::colon) || NextToken().is(tok::r_square))) {
        LHS = ParseObjCMessageExpressionBody(SourceLocation(), SourceLocation(),
                                             ParsedType(), LHS.get());
        break;
      }
        
      // Fall through; this isn't a message send.
                
    default:  // Not a postfix-expression suffix.
      return move(LHS);
    case tok::l_square: {  // postfix-expression: p-e '[' expression ']'
      // If we have a array postfix expression that starts on a new line and
      // Objective-C is enabled, it is highly likely that the user forgot a
      // semicolon after the base expression and that the array postfix-expr is
      // actually another message send.  In this case, do some look-ahead to see
      // if the contents of the square brackets are obviously not a valid
      // expression and recover by pretending there is no suffix.
      if (getLangOpts().ObjC1 && Tok.isAtStartOfLine() &&
          isSimpleObjCMessageExpression())
        return move(LHS);

      // Reject array indices starting with a lambda-expression. '[[' is
      // reserved for attributes.
      if (CheckProhibitedCXX11Attribute())
        return ExprError();

      BalancedDelimiterTracker T(*this, tok::l_square);
      T.consumeOpen();
      Loc = T.getOpenLocation();
      ExprResult Idx;
      if (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace)) {
        Diag(Tok, diag::warn_cxx98_compat_generalized_initializer_lists);
        Idx = ParseBraceInitializer();
      } else
        Idx = ParseExpression();

      SourceLocation RLoc = Tok.getLocation();

      if (!LHS.isInvalid() && !Idx.isInvalid() && Tok.is(tok::r_square)) {
        LHS = Actions.ActOnArraySubscriptExpr(getCurScope(), LHS.take(), Loc,
                                              Idx.take(), RLoc);
      } else
        LHS = ExprError();

      // Match the ']'.
      T.consumeClose();
      break;
    }

    case tok::l_paren:         // p-e: p-e '(' argument-expression-list[opt] ')'
    case tok::lesslessless: {  // p-e: p-e '<<<' argument-expression-list '>>>'
                               //   '(' argument-expression-list[opt] ')'
      tok::TokenKind OpKind = Tok.getKind();
      InMessageExpressionRAIIObject InMessage(*this, false);
      
      Expr *ExecConfig = 0;

      BalancedDelimiterTracker PT(*this, tok::l_paren);

      if (OpKind == tok::lesslessless) {
        ExprVector ExecConfigExprs(Actions);
        CommaLocsTy ExecConfigCommaLocs;
        SourceLocation OpenLoc = ConsumeToken();

        if (ParseExpressionList(ExecConfigExprs, ExecConfigCommaLocs)) {
          LHS = ExprError();
        }

        SourceLocation CloseLoc = Tok.getLocation();
        if (Tok.is(tok::greatergreatergreater)) {
          ConsumeToken();
        } else if (LHS.isInvalid()) {
          SkipUntil(tok::greatergreatergreater);
        } else {
          // There was an error closing the brackets
          Diag(Tok, diag::err_expected_ggg);
          Diag(OpenLoc, diag::note_matching) << "<<<";
          SkipUntil(tok::greatergreatergreater);
          LHS = ExprError();
        }

        if (!LHS.isInvalid()) {
          if (ExpectAndConsume(tok::l_paren, diag::err_expected_lparen, ""))
            LHS = ExprError();
          else
            Loc = PrevTokLocation;
        }

        if (!LHS.isInvalid()) {
          ExprResult ECResult = Actions.ActOnCUDAExecConfigExpr(getCurScope(),
                                    OpenLoc, 
                                    move_arg(ExecConfigExprs), 
                                    CloseLoc);
          if (ECResult.isInvalid())
            LHS = ExprError();
          else
            ExecConfig = ECResult.get();
        }
      } else {
        PT.consumeOpen();
        Loc = PT.getOpenLocation();
      }

      ExprVector ArgExprs(Actions);
      CommaLocsTy CommaLocs;
      
      if (Tok.is(tok::code_completion)) {
        Actions.CodeCompleteCall(getCurScope(), LHS.get(),
                                 llvm::ArrayRef<Expr *>());
        cutOffParsing();
        return ExprError();
      }

      if (OpKind == tok::l_paren || !LHS.isInvalid()) {
        if (Tok.isNot(tok::r_paren)) {
          if (ParseExpressionList(ArgExprs, CommaLocs, &Sema::CodeCompleteCall,
                                  LHS.get())) {
            LHS = ExprError();
          }
        }
      }

      // Match the ')'.
      if (LHS.isInvalid()) {
        SkipUntil(tok::r_paren);
      } else if (Tok.isNot(tok::r_paren)) {
        PT.consumeClose();
        LHS = ExprError();
      } else {
        assert((ArgExprs.size() == 0 || 
                ArgExprs.size()-1 == CommaLocs.size())&&
               "Unexpected number of commas!");
        LHS = Actions.ActOnCallExpr(getCurScope(), LHS.take(), Loc,
                                    move_arg(ArgExprs), Tok.getLocation(),
                                    ExecConfig);
        PT.consumeClose();
      }

      break;
    }
    case tok::arrow:
    case tok::period: {
      // postfix-expression: p-e '->' template[opt] id-expression
      // postfix-expression: p-e '.' template[opt] id-expression
      tok::TokenKind OpKind = Tok.getKind();
      SourceLocation OpLoc = ConsumeToken();  // Eat the "." or "->" token.

      CXXScopeSpec SS;
      ParsedType ObjectType;
      bool MayBePseudoDestructor = false;
      if (getLangOpts().CPlusPlus && !LHS.isInvalid()) {
        LHS = Actions.ActOnStartCXXMemberReference(getCurScope(), LHS.take(),
                                                   OpLoc, OpKind, ObjectType,
                                                   MayBePseudoDestructor);
        if (LHS.isInvalid())
          break;

        ParseOptionalCXXScopeSpecifier(SS, ObjectType, 
                                       /*EnteringContext=*/false,
                                       &MayBePseudoDestructor);
        if (SS.isNotEmpty())
          ObjectType = ParsedType();
      }

      if (Tok.is(tok::code_completion)) {
        // Code completion for a member access expression.
        Actions.CodeCompleteMemberReferenceExpr(getCurScope(), LHS.get(),
                                                OpLoc, OpKind == tok::arrow);
        
        cutOffParsing();
        return ExprError();
      }
      
      if (MayBePseudoDestructor && !LHS.isInvalid()) {
        LHS = ParseCXXPseudoDestructor(LHS.take(), OpLoc, OpKind, SS, 
                                       ObjectType);
        break;
      }

      // Either the action has told is that this cannot be a
      // pseudo-destructor expression (based on the type of base
      // expression), or we didn't see a '~' in the right place. We
      // can still parse a destructor name here, but in that case it
      // names a real destructor.
      // Allow explicit constructor calls in Microsoft mode.
      // FIXME: Add support for explicit call of template constructor.
      SourceLocation TemplateKWLoc;
      UnqualifiedId Name;
      if (getLangOpts().ObjC2 && OpKind == tok::period && Tok.is(tok::kw_class)) {
        // Objective-C++:
        //   After a '.' in a member access expression, treat the keyword
        //   'class' as if it were an identifier.
        //
        // This hack allows property access to the 'class' method because it is
        // such a common method name. For other C++ keywords that are 
        // Objective-C method names, one must use the message send syntax.
        IdentifierInfo *Id = Tok.getIdentifierInfo();
        SourceLocation Loc = ConsumeToken();
        Name.setIdentifier(Id, Loc);
      } else if (ParseUnqualifiedId(SS, 
                                    /*EnteringContext=*/false, 
                                    /*AllowDestructorName=*/true,
                                    /*AllowConstructorName=*/
                                      getLangOpts().MicrosoftExt, 
                                    ObjectType, TemplateKWLoc, Name))
        LHS = ExprError();
      
      if (!LHS.isInvalid())
        LHS = Actions.ActOnMemberAccessExpr(getCurScope(), LHS.take(), OpLoc, 
                                            OpKind, SS, TemplateKWLoc, Name,
                                 CurParsedObjCImpl ? CurParsedObjCImpl->Dcl : 0,
                                            Tok.is(tok::l_paren));
      break;
    }
    case tok::plusplus:    // postfix-expression: postfix-expression '++'
    case tok::minusminus:  // postfix-expression: postfix-expression '--'
      if (!LHS.isInvalid()) {
        LHS = Actions.ActOnPostfixUnaryOp(getCurScope(), Tok.getLocation(),
                                          Tok.getKind(), LHS.take());
      }
      ConsumeToken();
      break;
    }
  }
}

/// ParseExprAfterUnaryExprOrTypeTrait - We parsed a typeof/sizeof/alignof/
/// vec_step and we are at the start of an expression or a parenthesized
/// type-id. OpTok is the operand token (typeof/sizeof/alignof). Returns the
/// expression (isCastExpr == false) or the type (isCastExpr == true).
///
///       unary-expression:  [C99 6.5.3]
///         'sizeof' unary-expression
///         'sizeof' '(' type-name ')'
/// [GNU]   '__alignof' unary-expression
/// [GNU]   '__alignof' '(' type-name ')'
/// [C++0x] 'alignof' '(' type-id ')'
///
/// [GNU]   typeof-specifier:
///           typeof ( expressions )
///           typeof ( type-name )
/// [GNU/C++] typeof unary-expression
///
/// [OpenCL 1.1 6.11.12] vec_step built-in function:
///           vec_step ( expressions )
///           vec_step ( type-name )
///
ExprResult
Parser::ParseExprAfterUnaryExprOrTypeTrait(const Token &OpTok,
                                           bool &isCastExpr,
                                           ParsedType &CastTy,
                                           SourceRange &CastRange) {

  assert((OpTok.is(tok::kw_typeof)    || OpTok.is(tok::kw_sizeof) ||
          OpTok.is(tok::kw___alignof) || OpTok.is(tok::kw_alignof) ||
          OpTok.is(tok::kw_vec_step)) &&
          "Not a typeof/sizeof/alignof/vec_step expression!");

  ExprResult Operand;

  // If the operand doesn't start with an '(', it must be an expression.
  if (Tok.isNot(tok::l_paren)) {
    isCastExpr = false;
    if (OpTok.is(tok::kw_typeof) && !getLangOpts().CPlusPlus) {
      Diag(Tok,diag::err_expected_lparen_after_id) << OpTok.getIdentifierInfo();
      return ExprError();
    }

    Operand = ParseCastExpression(true/*isUnaryExpression*/);
  } else {
    // If it starts with a '(', we know that it is either a parenthesized
    // type-name, or it is a unary-expression that starts with a compound
    // literal, or starts with a primary-expression that is a parenthesized
    // expression.
    ParenParseOption ExprType = CastExpr;
    SourceLocation LParenLoc = Tok.getLocation(), RParenLoc;

    Operand = ParseParenExpression(ExprType, true/*stopIfCastExpr*/, 
                                   false, CastTy, RParenLoc);
    CastRange = SourceRange(LParenLoc, RParenLoc);

    // If ParseParenExpression parsed a '(typename)' sequence only, then this is
    // a type.
    if (ExprType == CastExpr) {
      isCastExpr = true;
      return ExprEmpty();
    }

    if (getLangOpts().CPlusPlus || OpTok.isNot(tok::kw_typeof)) {
      // GNU typeof in C requires the expression to be parenthesized. Not so for
      // sizeof/alignof or in C++. Therefore, the parenthesized expression is
      // the start of a unary-expression, but doesn't include any postfix 
      // pieces. Parse these now if present.
      if (!Operand.isInvalid())
        Operand = ParsePostfixExpressionSuffix(Operand.get());
    }
  }

  // If we get here, the operand to the typeof/sizeof/alignof was an expresion.
  isCastExpr = false;
  return move(Operand);
}


/// ParseUnaryExprOrTypeTraitExpression - Parse a sizeof or alignof expression.
///       unary-expression:  [C99 6.5.3]
///         'sizeof' unary-expression
///         'sizeof' '(' type-name ')'
/// [C++0x] 'sizeof' '...' '(' identifier ')'
/// [GNU]   '__alignof' unary-expression
/// [GNU]   '__alignof' '(' type-name ')'
/// [C++0x] 'alignof' '(' type-id ')'
ExprResult Parser::ParseUnaryExprOrTypeTraitExpression() {
  assert((Tok.is(tok::kw_sizeof) || Tok.is(tok::kw___alignof)
          || Tok.is(tok::kw_alignof) || Tok.is(tok::kw_vec_step)) &&
         "Not a sizeof/alignof/vec_step expression!");
  Token OpTok = Tok;
  ConsumeToken();

  // [C++0x] 'sizeof' '...' '(' identifier ')'
  if (Tok.is(tok::ellipsis) && OpTok.is(tok::kw_sizeof)) {
    SourceLocation EllipsisLoc = ConsumeToken();
    SourceLocation LParenLoc, RParenLoc;
    IdentifierInfo *Name = 0;
    SourceLocation NameLoc;
    if (Tok.is(tok::l_paren)) {
      BalancedDelimiterTracker T(*this, tok::l_paren);
      T.consumeOpen();
      LParenLoc = T.getOpenLocation();
      if (Tok.is(tok::identifier)) {
        Name = Tok.getIdentifierInfo();
        NameLoc = ConsumeToken();
        T.consumeClose();
        RParenLoc = T.getCloseLocation();
        if (RParenLoc.isInvalid())
          RParenLoc = PP.getLocForEndOfToken(NameLoc);
      } else {
        Diag(Tok, diag::err_expected_parameter_pack);
        SkipUntil(tok::r_paren);
      }
    } else if (Tok.is(tok::identifier)) {
      Name = Tok.getIdentifierInfo();
      NameLoc = ConsumeToken();
      LParenLoc = PP.getLocForEndOfToken(EllipsisLoc);
      RParenLoc = PP.getLocForEndOfToken(NameLoc);
      Diag(LParenLoc, diag::err_paren_sizeof_parameter_pack)
        << Name
        << FixItHint::CreateInsertion(LParenLoc, "(")
        << FixItHint::CreateInsertion(RParenLoc, ")");
    } else {
      Diag(Tok, diag::err_sizeof_parameter_pack);
    }
    
    if (!Name)
      return ExprError();
    
    return Actions.ActOnSizeofParameterPackExpr(getCurScope(),
                                                OpTok.getLocation(), 
                                                *Name, NameLoc,
                                                RParenLoc);
  }

  if (OpTok.is(tok::kw_alignof))
    Diag(OpTok, diag::warn_cxx98_compat_alignof);

  EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated);

  bool isCastExpr;
  ParsedType CastTy;
  SourceRange CastRange;
  ExprResult Operand = ParseExprAfterUnaryExprOrTypeTrait(OpTok,
                                                          isCastExpr,
                                                          CastTy,
                                                          CastRange);

  UnaryExprOrTypeTrait ExprKind = UETT_SizeOf;
  if (OpTok.is(tok::kw_alignof) || OpTok.is(tok::kw___alignof))
    ExprKind = UETT_AlignOf;
  else if (OpTok.is(tok::kw_vec_step))
    ExprKind = UETT_VecStep;

  if (isCastExpr)
    return Actions.ActOnUnaryExprOrTypeTraitExpr(OpTok.getLocation(),
                                                 ExprKind,
                                                 /*isType=*/true,
                                                 CastTy.getAsOpaquePtr(),
                                                 CastRange);

  // If we get here, the operand to the sizeof/alignof was an expresion.
  if (!Operand.isInvalid())
    Operand = Actions.ActOnUnaryExprOrTypeTraitExpr(OpTok.getLocation(),
                                                    ExprKind,
                                                    /*isType=*/false,
                                                    Operand.release(),
                                                    CastRange);
  return move(Operand);
}

/// ParseBuiltinPrimaryExpression
///
///       primary-expression: [C99 6.5.1]
/// [GNU]   '__builtin_va_arg' '(' assignment-expression ',' type-name ')'
/// [GNU]   '__builtin_offsetof' '(' type-name ',' offsetof-member-designator')'
/// [GNU]   '__builtin_choose_expr' '(' assign-expr ',' assign-expr ','
///                                     assign-expr ')'
/// [GNU]   '__builtin_types_compatible_p' '(' type-name ',' type-name ')'
/// [OCL]   '__builtin_astype' '(' assignment-expression ',' type-name ')'
///
/// [GNU] offsetof-member-designator:
/// [GNU]   identifier
/// [GNU]   offsetof-member-designator '.' identifier
/// [GNU]   offsetof-member-designator '[' expression ']'
///
ExprResult Parser::ParseBuiltinPrimaryExpression() {
  ExprResult Res;
  const IdentifierInfo *BuiltinII = Tok.getIdentifierInfo();

  tok::TokenKind T = Tok.getKind();
  SourceLocation StartLoc = ConsumeToken();   // Eat the builtin identifier.

  // All of these start with an open paren.
  if (Tok.isNot(tok::l_paren))
    return ExprError(Diag(Tok, diag::err_expected_lparen_after_id)
                       << BuiltinII);

  BalancedDelimiterTracker PT(*this, tok::l_paren);
  PT.consumeOpen();

  // TODO: Build AST.

  switch (T) {
  default: llvm_unreachable("Not a builtin primary expression!");
  case tok::kw___builtin_va_arg: {
    ExprResult Expr(ParseAssignmentExpression());

    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      Expr = ExprError();

    TypeResult Ty = ParseTypeName();

    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      Expr = ExprError();
    }

    if (Expr.isInvalid() || Ty.isInvalid())
      Res = ExprError();
    else
      Res = Actions.ActOnVAArg(StartLoc, Expr.take(), Ty.get(), ConsumeParen());
    break;
  }
  case tok::kw___builtin_offsetof: {
    SourceLocation TypeLoc = Tok.getLocation();
    TypeResult Ty = ParseTypeName();
    if (Ty.isInvalid()) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }

    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprError();

    // We must have at least one identifier here.
    if (Tok.isNot(tok::identifier)) {
      Diag(Tok, diag::err_expected_ident);
      SkipUntil(tok::r_paren);
      return ExprError();
    }

    // Keep track of the various subcomponents we see.
    SmallVector<Sema::OffsetOfComponent, 4> Comps;

    Comps.push_back(Sema::OffsetOfComponent());
    Comps.back().isBrackets = false;
    Comps.back().U.IdentInfo = Tok.getIdentifierInfo();
    Comps.back().LocStart = Comps.back().LocEnd = ConsumeToken();

    // FIXME: This loop leaks the index expressions on error.
    while (1) {
      if (Tok.is(tok::period)) {
        // offsetof-member-designator: offsetof-member-designator '.' identifier
        Comps.push_back(Sema::OffsetOfComponent());
        Comps.back().isBrackets = false;
        Comps.back().LocStart = ConsumeToken();

        if (Tok.isNot(tok::identifier)) {
          Diag(Tok, diag::err_expected_ident);
          SkipUntil(tok::r_paren);
          return ExprError();
        }
        Comps.back().U.IdentInfo = Tok.getIdentifierInfo();
        Comps.back().LocEnd = ConsumeToken();

      } else if (Tok.is(tok::l_square)) {
        if (CheckProhibitedCXX11Attribute())
          return ExprError();

        // offsetof-member-designator: offsetof-member-design '[' expression ']'
        Comps.push_back(Sema::OffsetOfComponent());
        Comps.back().isBrackets = true;
        BalancedDelimiterTracker ST(*this, tok::l_square);
        ST.consumeOpen();
        Comps.back().LocStart = ST.getOpenLocation();
        Res = ParseExpression();
        if (Res.isInvalid()) {
          SkipUntil(tok::r_paren);
          return move(Res);
        }
        Comps.back().U.E = Res.release();

        ST.consumeClose();
        Comps.back().LocEnd = ST.getCloseLocation();
      } else {
        if (Tok.isNot(tok::r_paren)) {
          PT.consumeClose();
          Res = ExprError();
        } else if (Ty.isInvalid()) {
          Res = ExprError();
        } else {
          PT.consumeClose();
          Res = Actions.ActOnBuiltinOffsetOf(getCurScope(), StartLoc, TypeLoc,
                                             Ty.get(), &Comps[0], Comps.size(),
                                             PT.getCloseLocation());
        }
        break;
      }
    }
    break;
  }
  case tok::kw___builtin_choose_expr: {
    ExprResult Cond(ParseAssignmentExpression());
    if (Cond.isInvalid()) {
      SkipUntil(tok::r_paren);
      return move(Cond);
    }
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprError();

    ExprResult Expr1(ParseAssignmentExpression());
    if (Expr1.isInvalid()) {
      SkipUntil(tok::r_paren);
      return move(Expr1);
    }
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "",tok::r_paren))
      return ExprError();

    ExprResult Expr2(ParseAssignmentExpression());
    if (Expr2.isInvalid()) {
      SkipUntil(tok::r_paren);
      return move(Expr2);
    }
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      return ExprError();
    }
    Res = Actions.ActOnChooseExpr(StartLoc, Cond.take(), Expr1.take(),
                                  Expr2.take(), ConsumeParen());
    break;
  }
  case tok::kw___builtin_astype: {
    // The first argument is an expression to be converted, followed by a comma.
    ExprResult Expr(ParseAssignmentExpression());
    if (Expr.isInvalid()) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }
    
    if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "", 
                         tok::r_paren))
      return ExprError();
    
    // Second argument is the type to bitcast to.
    TypeResult DestTy = ParseTypeName();
    if (DestTy.isInvalid())
      return ExprError();
    
    // Attempt to consume the r-paren.
    if (Tok.isNot(tok::r_paren)) {
      Diag(Tok, diag::err_expected_rparen);
      SkipUntil(tok::r_paren);
      return ExprError();
    }
    
    Res = Actions.ActOnAsTypeExpr(Expr.take(), DestTy.get(), StartLoc, 
                                  ConsumeParen());
    break;
  }
  }

  if (Res.isInvalid())
    return ExprError();

  // These can be followed by postfix-expr pieces because they are
  // primary-expressions.
  return ParsePostfixExpressionSuffix(Res.take());
}

/// ParseParenExpression - This parses the unit that starts with a '(' token,
/// based on what is allowed by ExprType.  The actual thing parsed is returned
/// in ExprType. If stopIfCastExpr is true, it will only return the parsed type,
/// not the parsed cast-expression.
///
///       primary-expression: [C99 6.5.1]
///         '(' expression ')'
/// [GNU]   '(' compound-statement ')'      (if !ParenExprOnly)
///       postfix-expression: [C99 6.5.2]
///         '(' type-name ')' '{' initializer-list '}'
///         '(' type-name ')' '{' initializer-list ',' '}'
///       cast-expression: [C99 6.5.4]
///         '(' type-name ')' cast-expression
/// [ARC]   bridged-cast-expression
/// 
/// [ARC] bridged-cast-expression:
///         (__bridge type-name) cast-expression
///         (__bridge_transfer type-name) cast-expression
///         (__bridge_retained type-name) cast-expression
ExprResult
Parser::ParseParenExpression(ParenParseOption &ExprType, bool stopIfCastExpr,
                             bool isTypeCast, ParsedType &CastTy,
                             SourceLocation &RParenLoc) {
  assert(Tok.is(tok::l_paren) && "Not a paren expr!");
  GreaterThanIsOperatorScope G(GreaterThanIsOperator, true);
  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.consumeOpen())
    return ExprError();
  SourceLocation OpenLoc = T.getOpenLocation();

  ExprResult Result(true);
  bool isAmbiguousTypeId;
  CastTy = ParsedType();

  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(), 
                 ExprType >= CompoundLiteral? Sema::PCC_ParenthesizedExpression
                                            : Sema::PCC_Expression);
    cutOffParsing();
    return ExprError();
  }

  // Diagnose use of bridge casts in non-arc mode.
  bool BridgeCast = (getLangOpts().ObjC2 &&
                     (Tok.is(tok::kw___bridge) || 
                      Tok.is(tok::kw___bridge_transfer) ||
                      Tok.is(tok::kw___bridge_retained) ||
                      Tok.is(tok::kw___bridge_retain)));
  if (BridgeCast && !getLangOpts().ObjCAutoRefCount) {
    StringRef BridgeCastName = Tok.getName();
    SourceLocation BridgeKeywordLoc = ConsumeToken();
    if (!PP.getSourceManager().isInSystemHeader(BridgeKeywordLoc))
      Diag(BridgeKeywordLoc, diag::warn_arc_bridge_cast_nonarc)
        << BridgeCastName
        << FixItHint::CreateReplacement(BridgeKeywordLoc, "");
    BridgeCast = false;
  }
  
  // None of these cases should fall through with an invalid Result
  // unless they've already reported an error.
  if (ExprType >= CompoundStmt && Tok.is(tok::l_brace)) {
    Diag(Tok, diag::ext_gnu_statement_expr);
    Actions.ActOnStartStmtExpr();

    StmtResult Stmt(ParseCompoundStatement(true));
    ExprType = CompoundStmt;

    // If the substmt parsed correctly, build the AST node.
    if (!Stmt.isInvalid()) {
      Result = Actions.ActOnStmtExpr(OpenLoc, Stmt.take(), Tok.getLocation());
    } else {
      Actions.ActOnStmtExprError();
    }
  } else if (ExprType >= CompoundLiteral && BridgeCast) {
    tok::TokenKind tokenKind = Tok.getKind();
    SourceLocation BridgeKeywordLoc = ConsumeToken();

    // Parse an Objective-C ARC ownership cast expression.
    ObjCBridgeCastKind Kind;
    if (tokenKind == tok::kw___bridge)
      Kind = OBC_Bridge;
    else if (tokenKind == tok::kw___bridge_transfer)
      Kind = OBC_BridgeTransfer;
    else if (tokenKind == tok::kw___bridge_retained)
      Kind = OBC_BridgeRetained;
    else {
      // As a hopefully temporary workaround, allow __bridge_retain as
      // a synonym for __bridge_retained, but only in system headers.
      assert(tokenKind == tok::kw___bridge_retain);
      Kind = OBC_BridgeRetained;
      if (!PP.getSourceManager().isInSystemHeader(BridgeKeywordLoc))
        Diag(BridgeKeywordLoc, diag::err_arc_bridge_retain)
          << FixItHint::CreateReplacement(BridgeKeywordLoc,
                                          "__bridge_retained");
    }
             
    TypeResult Ty = ParseTypeName();
    T.consumeClose();
    RParenLoc = T.getCloseLocation();
    ExprResult SubExpr = ParseCastExpression(/*isUnaryExpression=*/false);
    
    if (Ty.isInvalid() || SubExpr.isInvalid())
      return ExprError();
    
    return Actions.ActOnObjCBridgedCast(getCurScope(), OpenLoc, Kind,
                                        BridgeKeywordLoc, Ty.get(),
                                        RParenLoc, SubExpr.get());
  } else if (ExprType >= CompoundLiteral &&
             isTypeIdInParens(isAmbiguousTypeId)) {

    // Otherwise, this is a compound literal expression or cast expression.

    // In C++, if the type-id is ambiguous we disambiguate based on context.
    // If stopIfCastExpr is true the context is a typeof/sizeof/alignof
    // in which case we should treat it as type-id.
    // if stopIfCastExpr is false, we need to determine the context past the
    // parens, so we defer to ParseCXXAmbiguousParenExpression for that.
    if (isAmbiguousTypeId && !stopIfCastExpr) {
      ExprResult res = ParseCXXAmbiguousParenExpression(ExprType, CastTy, T);
      RParenLoc = T.getCloseLocation();
      return res;
    }

    // Parse the type declarator.
    DeclSpec DS(AttrFactory);
    ParseSpecifierQualifierList(DS);
    Declarator DeclaratorInfo(DS, Declarator::TypeNameContext);
    ParseDeclarator(DeclaratorInfo);
    
    // If our type is followed by an identifier and either ':' or ']', then 
    // this is probably an Objective-C message send where the leading '[' is
    // missing. Recover as if that were the case.
    if (!DeclaratorInfo.isInvalidType() && Tok.is(tok::identifier) &&
        !InMessageExpression && getLangOpts().ObjC1 &&
        (NextToken().is(tok::colon) || NextToken().is(tok::r_square))) {
      TypeResult Ty;
      {
        InMessageExpressionRAIIObject InMessage(*this, false);
        Ty = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
      }
      Result = ParseObjCMessageExpressionBody(SourceLocation(), 
                                              SourceLocation(), 
                                              Ty.get(), 0);
    } else {          
      // Match the ')'.
      T.consumeClose();
      RParenLoc = T.getCloseLocation();
      if (Tok.is(tok::l_brace)) {
        ExprType = CompoundLiteral;
        TypeResult Ty;
        {
          InMessageExpressionRAIIObject InMessage(*this, false);
          Ty = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
        }
        return ParseCompoundLiteralExpression(Ty.get(), OpenLoc, RParenLoc);
      }

      if (ExprType == CastExpr) {
        // We parsed '(' type-name ')' and the thing after it wasn't a '{'.

        if (DeclaratorInfo.isInvalidType())
          return ExprError();

        // Note that this doesn't parse the subsequent cast-expression, it just
        // returns the parsed type to the callee.
        if (stopIfCastExpr) {
          TypeResult Ty;
          {
            InMessageExpressionRAIIObject InMessage(*this, false);
            Ty = Actions.ActOnTypeName(getCurScope(), DeclaratorInfo);
          }
          CastTy = Ty.get();
          return ExprResult();
        }
        
        // Reject the cast of super idiom in ObjC.
        if (Tok.is(tok::identifier) && getLangOpts().ObjC1 &&
            Tok.getIdentifierInfo() == Ident_super && 
            getCurScope()->isInObjcMethodScope() &&
            GetLookAheadToken(1).isNot(tok::period)) {
          Diag(Tok.getLocation(), diag::err_illegal_super_cast)
            << SourceRange(OpenLoc, RParenLoc);
          return ExprError();
        }

        // Parse the cast-expression that follows it next.
        // TODO: For cast expression with CastTy.
        Result = ParseCastExpression(/*isUnaryExpression=*/false,
                                     /*isAddressOfOperand=*/false,
                                     /*isTypeCast=*/IsTypeCast);
        if (!Result.isInvalid()) {
          Result = Actions.ActOnCastExpr(getCurScope(), OpenLoc,
                                         DeclaratorInfo, CastTy, 
                                         RParenLoc, Result.take());
        }
        return move(Result);
      }

      Diag(Tok, diag::err_expected_lbrace_in_compound_literal);
      return ExprError();
    }
  } else if (isTypeCast) {
    // Parse the expression-list.
    InMessageExpressionRAIIObject InMessage(*this, false);
    
    ExprVector ArgExprs(Actions);
    CommaLocsTy CommaLocs;

    if (!ParseExpressionList(ArgExprs, CommaLocs)) {
      ExprType = SimpleExpr;
      Result = Actions.ActOnParenListExpr(OpenLoc, Tok.getLocation(),
                                          move_arg(ArgExprs));
    }
  } else {
    InMessageExpressionRAIIObject InMessage(*this, false);
    
    Result = ParseExpression(MaybeTypeCast);
    ExprType = SimpleExpr;

    // Don't build a paren expression unless we actually match a ')'.
    if (!Result.isInvalid() && Tok.is(tok::r_paren))
      Result = Actions.ActOnParenExpr(OpenLoc, Tok.getLocation(), Result.take());
  }

  // Match the ')'.
  if (Result.isInvalid()) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  T.consumeClose();
  RParenLoc = T.getCloseLocation();
  return move(Result);
}

/// ParseCompoundLiteralExpression - We have parsed the parenthesized type-name
/// and we are at the left brace.
///
///       postfix-expression: [C99 6.5.2]
///         '(' type-name ')' '{' initializer-list '}'
///         '(' type-name ')' '{' initializer-list ',' '}'
///
ExprResult
Parser::ParseCompoundLiteralExpression(ParsedType Ty,
                                       SourceLocation LParenLoc,
                                       SourceLocation RParenLoc) {
  assert(Tok.is(tok::l_brace) && "Not a compound literal!");
  if (!getLangOpts().C99)   // Compound literals don't exist in C90.
    Diag(LParenLoc, diag::ext_c99_compound_literal);
  ExprResult Result = ParseInitializer();
  if (!Result.isInvalid() && Ty)
    return Actions.ActOnCompoundLiteral(LParenLoc, Ty, RParenLoc, Result.take());
  return move(Result);
}

/// ParseStringLiteralExpression - This handles the various token types that
/// form string literals, and also handles string concatenation [C99 5.1.1.2,
/// translation phase #6].
///
///       primary-expression: [C99 6.5.1]
///         string-literal
ExprResult Parser::ParseStringLiteralExpression(bool AllowUserDefinedLiteral) {
  assert(isTokenStringLiteral() && "Not a string literal!");

  // String concat.  Note that keywords like __func__ and __FUNCTION__ are not
  // considered to be strings for concatenation purposes.
  SmallVector<Token, 4> StringToks;

  do {
    StringToks.push_back(Tok);
    ConsumeStringToken();
  } while (isTokenStringLiteral());

  // Pass the set of string tokens, ready for concatenation, to the actions.
  return Actions.ActOnStringLiteral(&StringToks[0], StringToks.size(),
                                   AllowUserDefinedLiteral ? getCurScope() : 0);
}

/// ParseGenericSelectionExpression - Parse a C11 generic-selection
/// [C11 6.5.1.1].
///
///    generic-selection:
///           _Generic ( assignment-expression , generic-assoc-list )
///    generic-assoc-list:
///           generic-association
///           generic-assoc-list , generic-association
///    generic-association:
///           type-name : assignment-expression
///           default : assignment-expression
ExprResult Parser::ParseGenericSelectionExpression() {
  assert(Tok.is(tok::kw__Generic) && "_Generic keyword expected");
  SourceLocation KeyLoc = ConsumeToken();

  if (!getLangOpts().C11)
    Diag(KeyLoc, diag::ext_c11_generic_selection);

  BalancedDelimiterTracker T(*this, tok::l_paren);
  if (T.expectAndConsume(diag::err_expected_lparen))
    return ExprError();

  ExprResult ControllingExpr;
  {
    // C11 6.5.1.1p3 "The controlling expression of a generic selection is
    // not evaluated."
    EnterExpressionEvaluationContext Unevaluated(Actions, Sema::Unevaluated);
    ControllingExpr = ParseAssignmentExpression();
    if (ControllingExpr.isInvalid()) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }
  }

  if (ExpectAndConsume(tok::comma, diag::err_expected_comma, "")) {
    SkipUntil(tok::r_paren);
    return ExprError();
  }

  SourceLocation DefaultLoc;
  TypeVector Types(Actions);
  ExprVector Exprs(Actions);
  while (1) {
    ParsedType Ty;
    if (Tok.is(tok::kw_default)) {
      // C11 6.5.1.1p2 "A generic selection shall have no more than one default
      // generic association."
      if (!DefaultLoc.isInvalid()) {
        Diag(Tok, diag::err_duplicate_default_assoc);
        Diag(DefaultLoc, diag::note_previous_default_assoc);
        SkipUntil(tok::r_paren);
        return ExprError();
      }
      DefaultLoc = ConsumeToken();
      Ty = ParsedType();
    } else {
      ColonProtectionRAIIObject X(*this);
      TypeResult TR = ParseTypeName();
      if (TR.isInvalid()) {
        SkipUntil(tok::r_paren);
        return ExprError();
      }
      Ty = TR.release();
    }
    Types.push_back(Ty);

    if (ExpectAndConsume(tok::colon, diag::err_expected_colon, "")) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }

    // FIXME: These expressions should be parsed in a potentially potentially
    // evaluated context.
    ExprResult ER(ParseAssignmentExpression());
    if (ER.isInvalid()) {
      SkipUntil(tok::r_paren);
      return ExprError();
    }
    Exprs.push_back(ER.release());

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken();
  }

  T.consumeClose();
  if (T.getCloseLocation().isInvalid())
    return ExprError();

  return Actions.ActOnGenericSelectionExpr(KeyLoc, DefaultLoc, 
                                           T.getCloseLocation(),
                                           ControllingExpr.release(),
                                           move_arg(Types), move_arg(Exprs));
}

/// ParseExpressionList - Used for C/C++ (argument-)expression-list.
///
///       argument-expression-list:
///         assignment-expression
///         argument-expression-list , assignment-expression
///
/// [C++] expression-list:
/// [C++]   assignment-expression
/// [C++]   expression-list , assignment-expression
///
/// [C++0x] expression-list:
/// [C++0x]   initializer-list
///
/// [C++0x] initializer-list
/// [C++0x]   initializer-clause ...[opt]
/// [C++0x]   initializer-list , initializer-clause ...[opt]
///
/// [C++0x] initializer-clause:
/// [C++0x]   assignment-expression
/// [C++0x]   braced-init-list
///
bool Parser::ParseExpressionList(SmallVectorImpl<Expr*> &Exprs,
                            SmallVectorImpl<SourceLocation> &CommaLocs,
                                 void (Sema::*Completer)(Scope *S, 
                                                           Expr *Data,
                                                   llvm::ArrayRef<Expr *> Args),
                                 Expr *Data) {
  while (1) {
    if (Tok.is(tok::code_completion)) {
      if (Completer)
        (Actions.*Completer)(getCurScope(), Data, Exprs);
      else
        Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Expression);
      cutOffParsing();
      return true;
    }

    ExprResult Expr;
    if (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace)) {
      Diag(Tok, diag::warn_cxx98_compat_generalized_initializer_lists);
      Expr = ParseBraceInitializer();
    } else
      Expr = ParseAssignmentExpression();

    if (Tok.is(tok::ellipsis))
      Expr = Actions.ActOnPackExpansion(Expr.get(), ConsumeToken());    
    if (Expr.isInvalid())
      return true;

    Exprs.push_back(Expr.release());

    if (Tok.isNot(tok::comma))
      return false;
    // Move to the next argument, remember where the comma was.
    CommaLocs.push_back(ConsumeToken());
  }
}

/// ParseBlockId - Parse a block-id, which roughly looks like int (int x).
///
/// [clang] block-id:
/// [clang]   specifier-qualifier-list block-declarator
///
void Parser::ParseBlockId() {
  if (Tok.is(tok::code_completion)) {
    Actions.CodeCompleteOrdinaryName(getCurScope(), Sema::PCC_Type);
    return cutOffParsing();
  }
  
  // Parse the specifier-qualifier-list piece.
  DeclSpec DS(AttrFactory);
  ParseSpecifierQualifierList(DS);

  // Parse the block-declarator.
  Declarator DeclaratorInfo(DS, Declarator::BlockLiteralContext);
  ParseDeclarator(DeclaratorInfo);

  // We do this for: ^ __attribute__((noreturn)) {, as DS has the attributes.
  DeclaratorInfo.takeAttributes(DS.getAttributes(), SourceLocation());

  MaybeParseGNUAttributes(DeclaratorInfo);

  // Inform sema that we are starting a block.
  Actions.ActOnBlockArguments(DeclaratorInfo, getCurScope());
}

/// ParseBlockLiteralExpression - Parse a block literal, which roughly looks
/// like ^(int x){ return x+1; }
///
///         block-literal:
/// [clang]   '^' block-args[opt] compound-statement
/// [clang]   '^' block-id compound-statement
/// [clang] block-args:
/// [clang]   '(' parameter-list ')'
///
ExprResult Parser::ParseBlockLiteralExpression() {
  assert(Tok.is(tok::caret) && "block literal starts with ^");
  SourceLocation CaretLoc = ConsumeToken();

  PrettyStackTraceLoc CrashInfo(PP.getSourceManager(), CaretLoc,
                                "block literal parsing");

  // Enter a scope to hold everything within the block.  This includes the
  // argument decls, decls within the compound expression, etc.  This also
  // allows determining whether a variable reference inside the block is
  // within or outside of the block.
  ParseScope BlockScope(this, Scope::BlockScope | Scope::FnScope |
                              Scope::DeclScope);

  // Inform sema that we are starting a block.
  Actions.ActOnBlockStart(CaretLoc, getCurScope());

  // Parse the return type if present.
  DeclSpec DS(AttrFactory);
  Declarator ParamInfo(DS, Declarator::BlockLiteralContext);
  // FIXME: Since the return type isn't actually parsed, it can't be used to
  // fill ParamInfo with an initial valid range, so do it manually.
  ParamInfo.SetSourceRange(SourceRange(Tok.getLocation(), Tok.getLocation()));

  // If this block has arguments, parse them.  There is no ambiguity here with
  // the expression case, because the expression case requires a parameter list.
  if (Tok.is(tok::l_paren)) {
    ParseParenDeclarator(ParamInfo);
    // Parse the pieces after the identifier as if we had "int(...)".
    // SetIdentifier sets the source range end, but in this case we're past
    // that location.
    SourceLocation Tmp = ParamInfo.getSourceRange().getEnd();
    ParamInfo.SetIdentifier(0, CaretLoc);
    ParamInfo.SetRangeEnd(Tmp);
    if (ParamInfo.isInvalidType()) {
      // If there was an error parsing the arguments, they may have
      // tried to use ^(x+y) which requires an argument list.  Just
      // skip the whole block literal.
      Actions.ActOnBlockError(CaretLoc, getCurScope());
      return ExprError();
    }

    MaybeParseGNUAttributes(ParamInfo);

    // Inform sema that we are starting a block.
    Actions.ActOnBlockArguments(ParamInfo, getCurScope());
  } else if (!Tok.is(tok::l_brace)) {
    ParseBlockId();
  } else {
    // Otherwise, pretend we saw (void).
    ParsedAttributes attrs(AttrFactory);
    ParamInfo.AddTypeInfo(DeclaratorChunk::getFunction(true, false,
                                                       SourceLocation(),
                                                       0, 0, 0,
                                                       true, SourceLocation(),
                                                       SourceLocation(),
                                                       SourceLocation(),
                                                       SourceLocation(),
                                                       EST_None,
                                                       SourceLocation(),
                                                       0, 0, 0, 0,
                                                       CaretLoc, CaretLoc,
                                                       ParamInfo),
                          attrs, CaretLoc);

    MaybeParseGNUAttributes(ParamInfo);

    // Inform sema that we are starting a block.
    Actions.ActOnBlockArguments(ParamInfo, getCurScope());
  }


  ExprResult Result(true);
  if (!Tok.is(tok::l_brace)) {
    // Saw something like: ^expr
    Diag(Tok, diag::err_expected_expression);
    Actions.ActOnBlockError(CaretLoc, getCurScope());
    return ExprError();
  }

  StmtResult Stmt(ParseCompoundStatementBody());
  BlockScope.Exit();
  if (!Stmt.isInvalid())
    Result = Actions.ActOnBlockStmtExpr(CaretLoc, Stmt.take(), getCurScope());
  else
    Actions.ActOnBlockError(CaretLoc, getCurScope());
  return move(Result);
}

/// ParseObjCBoolLiteral - This handles the objective-c Boolean literals.
///
///         '__objc_yes'
///         '__objc_no'
ExprResult Parser::ParseObjCBoolLiteral() {
  tok::TokenKind Kind = Tok.getKind();
  return Actions.ActOnObjCBoolLiteral(ConsumeToken(), Kind);
}
