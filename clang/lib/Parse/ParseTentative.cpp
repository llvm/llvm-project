//===--- ParseTentative.cpp - Ambiguity Resolution Parsing ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the tentative parsing portions of the Parser
//  interfaces, for ambiguity resolution.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/Parser.h"
#include "clang/Parse/ParseDiagnostic.h"
#include "clang/Sema/ParsedTemplate.h"
using namespace clang;

/// isCXXDeclarationStatement - C++-specialized function that disambiguates
/// between a declaration or an expression statement, when parsing function
/// bodies. Returns true for declaration, false for expression.
///
///         declaration-statement:
///           block-declaration
///
///         block-declaration:
///           simple-declaration
///           asm-definition
///           namespace-alias-definition
///           using-declaration
///           using-directive
/// [C++0x]   static_assert-declaration
///
///         asm-definition:
///           'asm' '(' string-literal ')' ';'
///
///         namespace-alias-definition:
///           'namespace' identifier = qualified-namespace-specifier ';'
///
///         using-declaration:
///           'using' typename[opt] '::'[opt] nested-name-specifier
///                 unqualified-id ';'
///           'using' '::' unqualified-id ;
///
///         using-directive:
///           'using' 'namespace' '::'[opt] nested-name-specifier[opt]
///                 namespace-name ';'
///
bool Parser::isCXXDeclarationStatement() {
  switch (Tok.getKind()) {
    // asm-definition
  case tok::kw_asm:
    // namespace-alias-definition
  case tok::kw_namespace:
    // using-declaration
    // using-directive
  case tok::kw_using:
    // static_assert-declaration
  case tok::kw_static_assert:
  case tok::kw__Static_assert:
    return true;
    // simple-declaration
  default:
    return isCXXSimpleDeclaration(/*AllowForRangeDecl=*/false);
  }
}

/// isCXXSimpleDeclaration - C++-specialized function that disambiguates
/// between a simple-declaration or an expression-statement.
/// If during the disambiguation process a parsing error is encountered,
/// the function returns true to let the declaration parsing code handle it.
/// Returns false if the statement is disambiguated as expression.
///
/// simple-declaration:
///   decl-specifier-seq init-declarator-list[opt] ';'
///
/// (if AllowForRangeDecl specified)
/// for ( for-range-declaration : for-range-initializer ) statement
/// for-range-declaration: 
///    attribute-specifier-seqopt type-specifier-seq declarator
bool Parser::isCXXSimpleDeclaration(bool AllowForRangeDecl) {
  // C++ 6.8p1:
  // There is an ambiguity in the grammar involving expression-statements and
  // declarations: An expression-statement with a function-style explicit type
  // conversion (5.2.3) as its leftmost subexpression can be indistinguishable
  // from a declaration where the first declarator starts with a '('. In those
  // cases the statement is a declaration. [Note: To disambiguate, the whole
  // statement might have to be examined to determine if it is an
  // expression-statement or a declaration].

  // C++ 6.8p3:
  // The disambiguation is purely syntactic; that is, the meaning of the names
  // occurring in such a statement, beyond whether they are type-names or not,
  // is not generally used in or changed by the disambiguation. Class
  // templates are instantiated as necessary to determine if a qualified name
  // is a type-name. Disambiguation precedes parsing, and a statement
  // disambiguated as a declaration may be an ill-formed declaration.

  // We don't have to parse all of the decl-specifier-seq part. There's only
  // an ambiguity if the first decl-specifier is
  // simple-type-specifier/typename-specifier followed by a '(', which may
  // indicate a function-style cast expression.
  // isCXXDeclarationSpecifier will return TPResult::Ambiguous() only in such
  // a case.

  bool InvalidAsDeclaration = false;
  TPResult TPR = isCXXDeclarationSpecifier(TPResult::False(),
                                           &InvalidAsDeclaration);
  if (TPR != TPResult::Ambiguous())
    return TPR != TPResult::False(); // Returns true for TPResult::True() or
                                     // TPResult::Error().

  // FIXME: TryParseSimpleDeclaration doesn't look past the first initializer,
  // and so gets some cases wrong. We can't carry on if we've already seen
  // something which makes this statement invalid as a declaration in this case,
  // since it can cause us to misparse valid code. Revisit this once
  // TryParseInitDeclaratorList is fixed.
  if (InvalidAsDeclaration)
    return false;

  // FIXME: Add statistics about the number of ambiguous statements encountered
  // and how they were resolved (number of declarations+number of expressions).

  // Ok, we have a simple-type-specifier/typename-specifier followed by a '(',
  // or an identifier which doesn't resolve as anything. We need tentative
  // parsing...

  TentativeParsingAction PA(*this);
  TPR = TryParseSimpleDeclaration(AllowForRangeDecl);
  PA.Revert();

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    return true;

  // Declarations take precedence over expressions.
  if (TPR == TPResult::Ambiguous())
    TPR = TPResult::True();

  assert(TPR == TPResult::True() || TPR == TPResult::False());
  return TPR == TPResult::True();
}

/// simple-declaration:
///   decl-specifier-seq init-declarator-list[opt] ';'
///
/// (if AllowForRangeDecl specified)
/// for ( for-range-declaration : for-range-initializer ) statement
/// for-range-declaration: 
///    attribute-specifier-seqopt type-specifier-seq declarator
///
Parser::TPResult Parser::TryParseSimpleDeclaration(bool AllowForRangeDecl) {
  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else {
    if (Tok.is(tok::annot_cxxscope))
      ConsumeToken();
    ConsumeToken();

    if (getLangOpts().ObjC1 && Tok.is(tok::less))
      TryParseProtocolQualifiers();
  }

  // Two decl-specifiers in a row conclusively disambiguate this as being a
  // simple-declaration. Don't bother calling isCXXDeclarationSpecifier in the
  // overwhelmingly common case that the next token is a '('.
  if (Tok.isNot(tok::l_paren)) {
    TPResult TPR = isCXXDeclarationSpecifier();
    if (TPR == TPResult::Ambiguous())
      return TPResult::True();
    if (TPR == TPResult::True() || TPR == TPResult::Error())
      return TPR;
    assert(TPR == TPResult::False());
  }

  TPResult TPR = TryParseInitDeclaratorList();
  if (TPR != TPResult::Ambiguous())
    return TPR;

  if (Tok.isNot(tok::semi) && (!AllowForRangeDecl || Tok.isNot(tok::colon)))
    return TPResult::False();

  return TPResult::Ambiguous();
}

///       init-declarator-list:
///         init-declarator
///         init-declarator-list ',' init-declarator
///
///       init-declarator:
///         declarator initializer[opt]
/// [GNU]   declarator simple-asm-expr[opt] attributes[opt] initializer[opt]
///
/// initializer:
///   '=' initializer-clause
///   '(' expression-list ')'
///
/// initializer-clause:
///   assignment-expression
///   '{' initializer-list ','[opt] '}'
///   '{' '}'
///
Parser::TPResult Parser::TryParseInitDeclaratorList() {
  while (1) {
    // declarator
    TPResult TPR = TryParseDeclarator(false/*mayBeAbstract*/);
    if (TPR != TPResult::Ambiguous())
      return TPR;

    // [GNU] simple-asm-expr[opt] attributes[opt]
    if (Tok.is(tok::kw_asm) || Tok.is(tok::kw___attribute))
      return TPResult::True();

    // initializer[opt]
    if (Tok.is(tok::l_paren)) {
      // Parse through the parens.
      ConsumeParen();
      if (!SkipUntil(tok::r_paren))
        return TPResult::Error();
    } else if (Tok.is(tok::equal) || isTokIdentifier_in()) {
      // MSVC and g++ won't examine the rest of declarators if '=' is 
      // encountered; they just conclude that we have a declaration.
      // EDG parses the initializer completely, which is the proper behavior
      // for this case.
      //
      // At present, Clang follows MSVC and g++, since the parser does not have
      // the ability to parse an expression fully without recording the
      // results of that parse.
      // Also allow 'in' after on objective-c declaration as in: 
      // for (int (^b)(void) in array). Ideally this should be done in the 
      // context of parsing for-init-statement of a foreach statement only. But,
      // in any other context 'in' is invalid after a declaration and parser
      // issues the error regardless of outcome of this decision.
      // FIXME. Change if above assumption does not hold.
      return TPResult::True();
    }

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // the comma.
  }

  return TPResult::Ambiguous();
}

/// isCXXConditionDeclaration - Disambiguates between a declaration or an
/// expression for a condition of a if/switch/while/for statement.
/// If during the disambiguation process a parsing error is encountered,
/// the function returns true to let the declaration parsing code handle it.
///
///       condition:
///         expression
///         type-specifier-seq declarator '=' assignment-expression
/// [C++11] type-specifier-seq declarator '=' initializer-clause
/// [C++11] type-specifier-seq declarator braced-init-list
/// [GNU]   type-specifier-seq declarator simple-asm-expr[opt] attributes[opt]
///             '=' assignment-expression
///
bool Parser::isCXXConditionDeclaration() {
  TPResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPResult::Ambiguous())
    return TPR != TPResult::False(); // Returns true for TPResult::True() or
                                     // TPResult::Error().

  // FIXME: Add statistics about the number of ambiguous statements encountered
  // and how they were resolved (number of declarations+number of expressions).

  // Ok, we have a simple-type-specifier/typename-specifier followed by a '('.
  // We need tentative parsing...

  TentativeParsingAction PA(*this);

  // type-specifier-seq
  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else {
    ConsumeToken();
    
    if (getLangOpts().ObjC1 && Tok.is(tok::less))
      TryParseProtocolQualifiers();
  }
  assert(Tok.is(tok::l_paren) && "Expected '('");

  // declarator
  TPR = TryParseDeclarator(false/*mayBeAbstract*/);

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    TPR = TPResult::True();

  if (TPR == TPResult::Ambiguous()) {
    // '='
    // [GNU] simple-asm-expr[opt] attributes[opt]
    if (Tok.is(tok::equal)  ||
        Tok.is(tok::kw_asm) || Tok.is(tok::kw___attribute))
      TPR = TPResult::True();
    else if (getLangOpts().CPlusPlus0x && Tok.is(tok::l_brace))
      TPR = TPResult::True();
    else
      TPR = TPResult::False();
  }

  PA.Revert();

  assert(TPR == TPResult::True() || TPR == TPResult::False());
  return TPR == TPResult::True();
}

  /// \brief Determine whether the next set of tokens contains a type-id.
  ///
  /// The context parameter states what context we're parsing right
  /// now, which affects how this routine copes with the token
  /// following the type-id. If the context is TypeIdInParens, we have
  /// already parsed the '(' and we will cease lookahead when we hit
  /// the corresponding ')'. If the context is
  /// TypeIdAsTemplateArgument, we've already parsed the '<' or ','
  /// before this template argument, and will cease lookahead when we
  /// hit a '>', '>>' (in C++0x), or ','. Returns true for a type-id
  /// and false for an expression.  If during the disambiguation
  /// process a parsing error is encountered, the function returns
  /// true to let the declaration parsing code handle it.
  ///
  /// type-id:
  ///   type-specifier-seq abstract-declarator[opt]
  ///
bool Parser::isCXXTypeId(TentativeCXXTypeIdContext Context, bool &isAmbiguous) {

  isAmbiguous = false;

  // C++ 8.2p2:
  // The ambiguity arising from the similarity between a function-style cast and
  // a type-id can occur in different contexts. The ambiguity appears as a
  // choice between a function-style cast expression and a declaration of a
  // type. The resolution is that any construct that could possibly be a type-id
  // in its syntactic context shall be considered a type-id.

  TPResult TPR = isCXXDeclarationSpecifier();
  if (TPR != TPResult::Ambiguous())
    return TPR != TPResult::False(); // Returns true for TPResult::True() or
                                     // TPResult::Error().

  // FIXME: Add statistics about the number of ambiguous statements encountered
  // and how they were resolved (number of declarations+number of expressions).

  // Ok, we have a simple-type-specifier/typename-specifier followed by a '('.
  // We need tentative parsing...

  TentativeParsingAction PA(*this);

  // type-specifier-seq
  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else {
    ConsumeToken();
    
    if (getLangOpts().ObjC1 && Tok.is(tok::less))
      TryParseProtocolQualifiers();
  }
  
  assert(Tok.is(tok::l_paren) && "Expected '('");

  // declarator
  TPR = TryParseDeclarator(true/*mayBeAbstract*/, false/*mayHaveIdentifier*/);

  // In case of an error, let the declaration parsing code handle it.
  if (TPR == TPResult::Error())
    TPR = TPResult::True();

  if (TPR == TPResult::Ambiguous()) {
    // We are supposed to be inside parens, so if after the abstract declarator
    // we encounter a ')' this is a type-id, otherwise it's an expression.
    if (Context == TypeIdInParens && Tok.is(tok::r_paren)) {
      TPR = TPResult::True();
      isAmbiguous = true;

    // We are supposed to be inside a template argument, so if after
    // the abstract declarator we encounter a '>', '>>' (in C++0x), or
    // ',', this is a type-id. Otherwise, it's an expression.
    } else if (Context == TypeIdAsTemplateArgument &&
               (Tok.is(tok::greater) || Tok.is(tok::comma) ||
                (getLangOpts().CPlusPlus0x && Tok.is(tok::greatergreater)))) {
      TPR = TPResult::True();
      isAmbiguous = true;

    } else
      TPR = TPResult::False();
  }

  PA.Revert();

  assert(TPR == TPResult::True() || TPR == TPResult::False());
  return TPR == TPResult::True();
}

/// \brief Returns true if this is a C++11 attribute-specifier. Per
/// C++11 [dcl.attr.grammar]p6, two consecutive left square bracket tokens
/// always introduce an attribute. In Objective-C++11, this rule does not
/// apply if either '[' begins a message-send.
///
/// If Disambiguate is true, we try harder to determine whether a '[[' starts
/// an attribute-specifier, and return CAK_InvalidAttributeSpecifier if not.
///
/// If OuterMightBeMessageSend is true, we assume the outer '[' is either an
/// Obj-C message send or the start of an attribute. Otherwise, we assume it
/// is not an Obj-C message send.
///
/// C++11 [dcl.attr.grammar]:
///
///     attribute-specifier:
///         '[' '[' attribute-list ']' ']'
///         alignment-specifier
///
///     attribute-list:
///         attribute[opt]
///         attribute-list ',' attribute[opt]
///         attribute '...'
///         attribute-list ',' attribute '...'
///
///     attribute:
///         attribute-token attribute-argument-clause[opt]
///
///     attribute-token:
///         identifier
///         identifier '::' identifier
///
///     attribute-argument-clause:
///         '(' balanced-token-seq ')'
Parser::CXX11AttributeKind
Parser::isCXX11AttributeSpecifier(bool Disambiguate,
                                  bool OuterMightBeMessageSend) {
  if (Tok.is(tok::kw_alignas))
    return CAK_AttributeSpecifier;

  if (Tok.isNot(tok::l_square) || NextToken().isNot(tok::l_square))
    return CAK_NotAttributeSpecifier;

  // No tentative parsing if we don't need to look for ']]' or a lambda.
  if (!Disambiguate && !getLangOpts().ObjC1)
    return CAK_AttributeSpecifier;

  TentativeParsingAction PA(*this);

  // Opening brackets were checked for above.
  ConsumeBracket();

  // Outside Obj-C++11, treat anything with a matching ']]' as an attribute.
  if (!getLangOpts().ObjC1) {
    ConsumeBracket();

    bool IsAttribute = SkipUntil(tok::r_square, false);
    IsAttribute &= Tok.is(tok::r_square);

    PA.Revert();

    return IsAttribute ? CAK_AttributeSpecifier : CAK_InvalidAttributeSpecifier;
  }

  // In Obj-C++11, we need to distinguish four situations:
  //  1a) int x[[attr]];                     C++11 attribute.
  //  1b) [[attr]];                          C++11 statement attribute.
  //   2) int x[[obj](){ return 1; }()];     Lambda in array size/index.
  //  3a) int x[[obj get]];                  Message send in array size/index.
  //  3b) [[Class alloc] init];              Message send in message send.
  //   4) [[obj]{ return self; }() doStuff]; Lambda in message send.
  // (1) is an attribute, (2) is ill-formed, and (3) and (4) are accepted.

  // If we have a lambda-introducer, then this is definitely not a message send.
  // FIXME: If this disambiguation is too slow, fold the tentative lambda parse
  // into the tentative attribute parse below.
  LambdaIntroducer Intro;
  if (!TryParseLambdaIntroducer(Intro)) {
    // A lambda cannot end with ']]', and an attribute must.
    bool IsAttribute = Tok.is(tok::r_square);

    PA.Revert();

    if (IsAttribute)
      // Case 1: C++11 attribute.
      return CAK_AttributeSpecifier;

    if (OuterMightBeMessageSend)
      // Case 4: Lambda in message send.
      return CAK_NotAttributeSpecifier;

    // Case 2: Lambda in array size / index.
    return CAK_InvalidAttributeSpecifier;
  }

  ConsumeBracket();

  // If we don't have a lambda-introducer, then we have an attribute or a
  // message-send.
  bool IsAttribute = true;
  while (Tok.isNot(tok::r_square)) {
    if (Tok.is(tok::comma)) {
      // Case 1: Stray commas can only occur in attributes.
      PA.Revert();
      return CAK_AttributeSpecifier;
    }

    // Parse the attribute-token, if present.
    // C++11 [dcl.attr.grammar]:
    //   If a keyword or an alternative token that satisfies the syntactic
    //   requirements of an identifier is contained in an attribute-token,
    //   it is considered an identifier.
    SourceLocation Loc;
    if (!TryParseCXX11AttributeIdentifier(Loc)) {
      IsAttribute = false;
      break;
    }
    if (Tok.is(tok::coloncolon)) {
      ConsumeToken();
      if (!TryParseCXX11AttributeIdentifier(Loc)) {
        IsAttribute = false;
        break;
      }
    }

    // Parse the attribute-argument-clause, if present.
    if (Tok.is(tok::l_paren)) {
      ConsumeParen();
      if (!SkipUntil(tok::r_paren, false)) {
        IsAttribute = false;
        break;
      }
    }

    if (Tok.is(tok::ellipsis))
      ConsumeToken();

    if (Tok.isNot(tok::comma))
      break;

    ConsumeToken();
  }

  // An attribute must end ']]'.
  if (IsAttribute) {
    if (Tok.is(tok::r_square)) {
      ConsumeBracket();
      IsAttribute = Tok.is(tok::r_square);
    } else {
      IsAttribute = false;
    }
  }

  PA.Revert();

  if (IsAttribute)
    // Case 1: C++11 statement attribute.
    return CAK_AttributeSpecifier;

  // Case 3: Message send.
  return CAK_NotAttributeSpecifier;
}

///         declarator:
///           direct-declarator
///           ptr-operator declarator
///
///         direct-declarator:
///           declarator-id
///           direct-declarator '(' parameter-declaration-clause ')'
///                 cv-qualifier-seq[opt] exception-specification[opt]
///           direct-declarator '[' constant-expression[opt] ']'
///           '(' declarator ')'
/// [GNU]     '(' attributes declarator ')'
///
///         abstract-declarator:
///           ptr-operator abstract-declarator[opt]
///           direct-abstract-declarator
///           ...
///
///         direct-abstract-declarator:
///           direct-abstract-declarator[opt]
///           '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///                 exception-specification[opt]
///           direct-abstract-declarator[opt] '[' constant-expression[opt] ']'
///           '(' abstract-declarator ')'
///
///         ptr-operator:
///           '*' cv-qualifier-seq[opt]
///           '&'
/// [C++0x]   '&&'                                                        [TODO]
///           '::'[opt] nested-name-specifier '*' cv-qualifier-seq[opt]
///
///         cv-qualifier-seq:
///           cv-qualifier cv-qualifier-seq[opt]
///
///         cv-qualifier:
///           'const'
///           'volatile'
///
///         declarator-id:
///           '...'[opt] id-expression
///
///         id-expression:
///           unqualified-id
///           qualified-id                                                [TODO]
///
///         unqualified-id:
///           identifier
///           operator-function-id                                        [TODO]
///           conversion-function-id                                      [TODO]
///           '~' class-name                                              [TODO]
///           template-id                                                 [TODO]
///
Parser::TPResult Parser::TryParseDeclarator(bool mayBeAbstract,
                                            bool mayHaveIdentifier) {
  // declarator:
  //   direct-declarator
  //   ptr-operator declarator

  while (1) {
    if (Tok.is(tok::coloncolon) || Tok.is(tok::identifier))
      if (TryAnnotateCXXScopeToken(true))
        return TPResult::Error();

    if (Tok.is(tok::star) || Tok.is(tok::amp) || Tok.is(tok::caret) ||
        Tok.is(tok::ampamp) ||
        (Tok.is(tok::annot_cxxscope) && NextToken().is(tok::star))) {
      // ptr-operator
      ConsumeToken();
      while (Tok.is(tok::kw_const)    ||
             Tok.is(tok::kw_volatile) ||
             Tok.is(tok::kw_restrict))
        ConsumeToken();
    } else {
      break;
    }
  }

  // direct-declarator:
  // direct-abstract-declarator:
  if (Tok.is(tok::ellipsis))
    ConsumeToken();
  
  if ((Tok.is(tok::identifier) ||
       (Tok.is(tok::annot_cxxscope) && NextToken().is(tok::identifier))) &&
      mayHaveIdentifier) {
    // declarator-id
    if (Tok.is(tok::annot_cxxscope))
      ConsumeToken();
    else
      TentativelyDeclaredIdentifiers.push_back(Tok.getIdentifierInfo());
    ConsumeToken();
  } else if (Tok.is(tok::l_paren)) {
    ConsumeParen();
    if (mayBeAbstract &&
        (Tok.is(tok::r_paren) ||       // 'int()' is a function.
         // 'int(...)' is a function.
         (Tok.is(tok::ellipsis) && NextToken().is(tok::r_paren)) ||
         isDeclarationSpecifier())) {   // 'int(int)' is a function.
      // '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
      //        exception-specification[opt]
      TPResult TPR = TryParseFunctionDeclarator();
      if (TPR != TPResult::Ambiguous())
        return TPR;
    } else {
      // '(' declarator ')'
      // '(' attributes declarator ')'
      // '(' abstract-declarator ')'
      if (Tok.is(tok::kw___attribute) ||
          Tok.is(tok::kw___declspec) ||
          Tok.is(tok::kw___cdecl) ||
          Tok.is(tok::kw___stdcall) ||
          Tok.is(tok::kw___fastcall) ||
          Tok.is(tok::kw___thiscall) ||
          Tok.is(tok::kw___unaligned))
        return TPResult::True(); // attributes indicate declaration
      TPResult TPR = TryParseDeclarator(mayBeAbstract, mayHaveIdentifier);
      if (TPR != TPResult::Ambiguous())
        return TPR;
      if (Tok.isNot(tok::r_paren))
        return TPResult::False();
      ConsumeParen();
    }
  } else if (!mayBeAbstract) {
    return TPResult::False();
  }

  while (1) {
    TPResult TPR(TPResult::Ambiguous());

    // abstract-declarator: ...
    if (Tok.is(tok::ellipsis))
      ConsumeToken();

    if (Tok.is(tok::l_paren)) {
      // Check whether we have a function declarator or a possible ctor-style
      // initializer that follows the declarator. Note that ctor-style
      // initializers are not possible in contexts where abstract declarators
      // are allowed.
      if (!mayBeAbstract && !isCXXFunctionDeclarator())
        break;

      // direct-declarator '(' parameter-declaration-clause ')'
      //        cv-qualifier-seq[opt] exception-specification[opt]
      ConsumeParen();
      TPR = TryParseFunctionDeclarator();
    } else if (Tok.is(tok::l_square)) {
      // direct-declarator '[' constant-expression[opt] ']'
      // direct-abstract-declarator[opt] '[' constant-expression[opt] ']'
      TPR = TryParseBracketDeclarator();
    } else {
      break;
    }

    if (TPR != TPResult::Ambiguous())
      return TPR;
  }

  return TPResult::Ambiguous();
}

Parser::TPResult 
Parser::isExpressionOrTypeSpecifierSimple(tok::TokenKind Kind) {
  switch (Kind) {
  // Obviously starts an expression.
  case tok::numeric_constant:
  case tok::char_constant:
  case tok::wide_char_constant:
  case tok::utf16_char_constant:
  case tok::utf32_char_constant:
  case tok::string_literal:
  case tok::wide_string_literal:
  case tok::utf8_string_literal:
  case tok::utf16_string_literal:
  case tok::utf32_string_literal:
  case tok::l_square:
  case tok::l_paren:
  case tok::amp:
  case tok::ampamp:
  case tok::star:
  case tok::plus:
  case tok::plusplus:
  case tok::minus:
  case tok::minusminus:
  case tok::tilde:
  case tok::exclaim:
  case tok::kw_sizeof:
  case tok::kw___func__:
  case tok::kw_const_cast:
  case tok::kw_delete:
  case tok::kw_dynamic_cast:
  case tok::kw_false:
  case tok::kw_new:
  case tok::kw_operator:
  case tok::kw_reinterpret_cast:
  case tok::kw_static_cast:
  case tok::kw_this:
  case tok::kw_throw:
  case tok::kw_true:
  case tok::kw_typeid:
  case tok::kw_alignof:
  case tok::kw_noexcept:
  case tok::kw_nullptr:
  case tok::kw__Alignof:
  case tok::kw___null:
  case tok::kw___alignof:
  case tok::kw___builtin_choose_expr:
  case tok::kw___builtin_offsetof:
  case tok::kw___builtin_types_compatible_p:
  case tok::kw___builtin_va_arg:
  case tok::kw___imag:
  case tok::kw___real:
  case tok::kw___FUNCTION__:
  case tok::kw_L__FUNCTION__:
  case tok::kw___PRETTY_FUNCTION__:
  case tok::kw___has_nothrow_assign:
  case tok::kw___has_nothrow_copy:
  case tok::kw___has_nothrow_constructor:
  case tok::kw___has_trivial_assign:
  case tok::kw___has_trivial_copy:
  case tok::kw___has_trivial_constructor:
  case tok::kw___has_trivial_destructor:
  case tok::kw___has_virtual_destructor:
  case tok::kw___is_abstract:
  case tok::kw___is_base_of:
  case tok::kw___is_class:
  case tok::kw___is_convertible_to:
  case tok::kw___is_empty:
  case tok::kw___is_enum:
  case tok::kw___is_interface_class:
  case tok::kw___is_final:
  case tok::kw___is_literal:
  case tok::kw___is_literal_type:
  case tok::kw___is_pod:
  case tok::kw___is_polymorphic:
  case tok::kw___is_trivial:
  case tok::kw___is_trivially_assignable:
  case tok::kw___is_trivially_constructible:
  case tok::kw___is_trivially_copyable:
  case tok::kw___is_union:
  case tok::kw___uuidof:
    return TPResult::True();
      
  // Obviously starts a type-specifier-seq:
  case tok::kw_char:
  case tok::kw_const:
  case tok::kw_double:
  case tok::kw_enum:
  case tok::kw_half:
  case tok::kw_float:
  case tok::kw_int:
  case tok::kw_long:
  case tok::kw___int64:
  case tok::kw___int128:
  case tok::kw_restrict:
  case tok::kw_short:
  case tok::kw_signed:
  case tok::kw_struct:
  case tok::kw_union:
  case tok::kw_unsigned:
  case tok::kw_void:
  case tok::kw_volatile:
  case tok::kw__Bool:
  case tok::kw__Complex:
  case tok::kw_class:
  case tok::kw_typename:
  case tok::kw_wchar_t:
  case tok::kw_char16_t:
  case tok::kw_char32_t:
  case tok::kw___underlying_type:
  case tok::kw_thread_local:
  case tok::kw__Decimal32:
  case tok::kw__Decimal64:
  case tok::kw__Decimal128:
  case tok::kw___thread:
  case tok::kw_typeof:
  case tok::kw___cdecl:
  case tok::kw___stdcall:
  case tok::kw___fastcall:
  case tok::kw___thiscall:
  case tok::kw___unaligned:
  case tok::kw___vector:
  case tok::kw___pixel:
  case tok::kw__Atomic:
    return TPResult::False();

  default:
    break;
  }
  
  return TPResult::Ambiguous();
}

bool Parser::isTentativelyDeclared(IdentifierInfo *II) {
  return std::find(TentativelyDeclaredIdentifiers.begin(),
                   TentativelyDeclaredIdentifiers.end(), II)
      != TentativelyDeclaredIdentifiers.end();
}

/// isCXXDeclarationSpecifier - Returns TPResult::True() if it is a declaration
/// specifier, TPResult::False() if it is not, TPResult::Ambiguous() if it could
/// be either a decl-specifier or a function-style cast, and TPResult::Error()
/// if a parsing error was found and reported.
///
/// If HasMissingTypename is provided, a name with a dependent scope specifier
/// will be treated as ambiguous if the 'typename' keyword is missing. If this
/// happens, *HasMissingTypename will be set to 'true'. This will also be used
/// as an indicator that undeclared identifiers (which will trigger a later
/// parse error) should be treated as types. Returns TPResult::Ambiguous() in
/// such cases.
///
///         decl-specifier:
///           storage-class-specifier
///           type-specifier
///           function-specifier
///           'friend'
///           'typedef'
/// [C++0x]   'constexpr'
/// [GNU]     attributes declaration-specifiers[opt]
///
///         storage-class-specifier:
///           'register'
///           'static'
///           'extern'
///           'mutable'
///           'auto'
/// [GNU]     '__thread'
///
///         function-specifier:
///           'inline'
///           'virtual'
///           'explicit'
///
///         typedef-name:
///           identifier
///
///         type-specifier:
///           simple-type-specifier
///           class-specifier
///           enum-specifier
///           elaborated-type-specifier
///           typename-specifier
///           cv-qualifier
///
///         simple-type-specifier:
///           '::'[opt] nested-name-specifier[opt] type-name
///           '::'[opt] nested-name-specifier 'template'
///                 simple-template-id                              [TODO]
///           'char'
///           'wchar_t'
///           'bool'
///           'short'
///           'int'
///           'long'
///           'signed'
///           'unsigned'
///           'float'
///           'double'
///           'void'
/// [GNU]     typeof-specifier
/// [GNU]     '_Complex'
/// [C++0x]   'auto'                                                [TODO]
/// [C++0x]   'decltype' ( expression )
///
///         type-name:
///           class-name
///           enum-name
///           typedef-name
///
///         elaborated-type-specifier:
///           class-key '::'[opt] nested-name-specifier[opt] identifier
///           class-key '::'[opt] nested-name-specifier[opt] 'template'[opt]
///               simple-template-id
///           'enum' '::'[opt] nested-name-specifier[opt] identifier
///
///         enum-name:
///           identifier
///
///         enum-specifier:
///           'enum' identifier[opt] '{' enumerator-list[opt] '}'
///           'enum' identifier[opt] '{' enumerator-list ',' '}'
///
///         class-specifier:
///           class-head '{' member-specification[opt] '}'
///
///         class-head:
///           class-key identifier[opt] base-clause[opt]
///           class-key nested-name-specifier identifier base-clause[opt]
///           class-key nested-name-specifier[opt] simple-template-id
///               base-clause[opt]
///
///         class-key:
///           'class'
///           'struct'
///           'union'
///
///         cv-qualifier:
///           'const'
///           'volatile'
/// [GNU]     restrict
///
Parser::TPResult
Parser::isCXXDeclarationSpecifier(Parser::TPResult BracedCastResult,
                                  bool *HasMissingTypename) {
  switch (Tok.getKind()) {
  case tok::identifier: {
    // Check for need to substitute AltiVec __vector keyword
    // for "vector" identifier.
    if (TryAltiVecVectorToken())
      return TPResult::True();

    const Token &Next = NextToken();
    // In 'foo bar', 'foo' is always a type name outside of Objective-C.
    if (!getLangOpts().ObjC1 && Next.is(tok::identifier))
      return TPResult::True();

    if (Next.isNot(tok::coloncolon) && Next.isNot(tok::less)) {
      // Determine whether this is a valid expression. If not, we will hit
      // a parse error one way or another. In that case, tell the caller that
      // this is ambiguous. Typo-correct to type and expression keywords and
      // to types and identifiers, in order to try to recover from errors.
      CorrectionCandidateCallback TypoCorrection;
      TypoCorrection.WantRemainingKeywords = false;
      switch (TryAnnotateName(false /* no nested name specifier */,
                              &TypoCorrection)) {
      case ANK_Error:
        return TPResult::Error();
      case ANK_TentativeDecl:
        return TPResult::False();
      case ANK_TemplateName:
        // A bare type template-name which can't be a template template
        // argument is an error, and was probably intended to be a type.
        return GreaterThanIsOperator ? TPResult::True() : TPResult::False();
      case ANK_Unresolved:
        return HasMissingTypename ? TPResult::Ambiguous() : TPResult::False();
      case ANK_Success:
        break;
      }
      assert(Tok.isNot(tok::identifier) &&
             "TryAnnotateName succeeded without producing an annotation");
    } else {
      // This might possibly be a type with a dependent scope specifier and
      // a missing 'typename' keyword. Don't use TryAnnotateName in this case,
      // since it will annotate as a primary expression, and we want to use the
      // "missing 'typename'" logic.
      if (TryAnnotateTypeOrScopeToken())
        return TPResult::Error();
      // If annotation failed, assume it's a non-type.
      // FIXME: If this happens due to an undeclared identifier, treat it as
      // ambiguous.
      if (Tok.is(tok::identifier))
        return TPResult::False();
    }

    // We annotated this token as something. Recurse to handle whatever we got.
    return isCXXDeclarationSpecifier(BracedCastResult, HasMissingTypename);
  }

  case tok::kw_typename:  // typename T::type
    // Annotate typenames and C++ scope specifiers.  If we get one, just
    // recurse to handle whatever we get.
    if (TryAnnotateTypeOrScopeToken())
      return TPResult::Error();
    return isCXXDeclarationSpecifier(BracedCastResult, HasMissingTypename);

  case tok::coloncolon: {    // ::foo::bar
    const Token &Next = NextToken();
    if (Next.is(tok::kw_new) ||    // ::new
        Next.is(tok::kw_delete))   // ::delete
      return TPResult::False();
  }
    // Fall through.
  case tok::kw_decltype:
    // Annotate typenames and C++ scope specifiers.  If we get one, just
    // recurse to handle whatever we get.
    if (TryAnnotateTypeOrScopeToken())
      return TPResult::Error();
    return isCXXDeclarationSpecifier(BracedCastResult, HasMissingTypename);

    // decl-specifier:
    //   storage-class-specifier
    //   type-specifier
    //   function-specifier
    //   'friend'
    //   'typedef'
    //   'constexpr'
  case tok::kw_friend:
  case tok::kw_typedef:
  case tok::kw_constexpr:
    // storage-class-specifier
  case tok::kw_register:
  case tok::kw_static:
  case tok::kw_extern:
  case tok::kw_mutable:
  case tok::kw_auto:
  case tok::kw___thread:
    // function-specifier
  case tok::kw_inline:
  case tok::kw_virtual:
  case tok::kw_explicit:

    // Modules
  case tok::kw___module_private__:
      
    // type-specifier:
    //   simple-type-specifier
    //   class-specifier
    //   enum-specifier
    //   elaborated-type-specifier
    //   typename-specifier
    //   cv-qualifier

    // class-specifier
    // elaborated-type-specifier
  case tok::kw_class:
  case tok::kw_struct:
  case tok::kw_union:
    // enum-specifier
  case tok::kw_enum:
    // cv-qualifier
  case tok::kw_const:
  case tok::kw_volatile:

    // GNU
  case tok::kw_restrict:
  case tok::kw__Complex:
  case tok::kw___attribute:
    return TPResult::True();

    // Microsoft
  case tok::kw___declspec:
  case tok::kw___cdecl:
  case tok::kw___stdcall:
  case tok::kw___fastcall:
  case tok::kw___thiscall:
  case tok::kw___w64:
  case tok::kw___ptr64:
  case tok::kw___ptr32:
  case tok::kw___forceinline:
  case tok::kw___unaligned:
    return TPResult::True();

    // Borland
  case tok::kw___pascal:
    return TPResult::True();
  
    // AltiVec
  case tok::kw___vector:
    return TPResult::True();

  case tok::annot_template_id: {
    TemplateIdAnnotation *TemplateId = takeTemplateIdAnnotation(Tok);
    if (TemplateId->Kind != TNK_Type_template)
      return TPResult::False();
    CXXScopeSpec SS;
    AnnotateTemplateIdTokenAsType();
    assert(Tok.is(tok::annot_typename));
    goto case_typename;
  }

  case tok::annot_cxxscope: // foo::bar or ::foo::bar, but already parsed
    // We've already annotated a scope; try to annotate a type.
    if (TryAnnotateTypeOrScopeToken())
      return TPResult::Error();
    if (!Tok.is(tok::annot_typename)) {
      // If the next token is an identifier or a type qualifier, then this
      // can't possibly be a valid expression either.
      if (Tok.is(tok::annot_cxxscope) && NextToken().is(tok::identifier)) {
        CXXScopeSpec SS;
        Actions.RestoreNestedNameSpecifierAnnotation(Tok.getAnnotationValue(),
                                                     Tok.getAnnotationRange(),
                                                     SS);
        if (SS.getScopeRep() && SS.getScopeRep()->isDependent()) {
          TentativeParsingAction PA(*this);
          ConsumeToken();
          ConsumeToken();
          bool isIdentifier = Tok.is(tok::identifier);
          TPResult TPR = TPResult::False();
          if (!isIdentifier)
            TPR = isCXXDeclarationSpecifier(BracedCastResult,
                                            HasMissingTypename);
          PA.Revert();

          if (isIdentifier ||
              TPR == TPResult::True() || TPR == TPResult::Error())
            return TPResult::Error();

          if (HasMissingTypename) {
            // We can't tell whether this is a missing 'typename' or a valid
            // expression.
            *HasMissingTypename = true;
            return TPResult::Ambiguous();
          }
        } else {
          // Try to resolve the name. If it doesn't exist, assume it was
          // intended to name a type and keep disambiguating.
          switch (TryAnnotateName(false /* SS is not dependent */)) {
          case ANK_Error:
            return TPResult::Error();
          case ANK_TentativeDecl:
            return TPResult::False();
          case ANK_TemplateName:
            // A bare type template-name which can't be a template template
            // argument is an error, and was probably intended to be a type.
            return GreaterThanIsOperator ? TPResult::True() : TPResult::False();
          case ANK_Unresolved:
            return HasMissingTypename ? TPResult::Ambiguous()
                                      : TPResult::False();
          case ANK_Success:
            // Annotated it, check again.
            assert(Tok.isNot(tok::annot_cxxscope) ||
                   NextToken().isNot(tok::identifier));
            return isCXXDeclarationSpecifier(BracedCastResult,
                                             HasMissingTypename);
          }
        }
      }
      return TPResult::False();
    }
    // If that succeeded, fallthrough into the generic simple-type-id case.

    // The ambiguity resides in a simple-type-specifier/typename-specifier
    // followed by a '('. The '(' could either be the start of:
    //
    //   direct-declarator:
    //     '(' declarator ')'
    //
    //   direct-abstract-declarator:
    //     '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
    //              exception-specification[opt]
    //     '(' abstract-declarator ')'
    //
    // or part of a function-style cast expression:
    //
    //     simple-type-specifier '(' expression-list[opt] ')'
    //

    // simple-type-specifier:

  case tok::annot_typename:
  case_typename:
    // In Objective-C, we might have a protocol-qualified type.
    if (getLangOpts().ObjC1 && NextToken().is(tok::less)) {
      // Tentatively parse the 
      TentativeParsingAction PA(*this);
      ConsumeToken(); // The type token
      
      TPResult TPR = TryParseProtocolQualifiers();
      bool isFollowedByParen = Tok.is(tok::l_paren);
      bool isFollowedByBrace = Tok.is(tok::l_brace);
      
      PA.Revert();
      
      if (TPR == TPResult::Error())
        return TPResult::Error();
      
      if (isFollowedByParen)
        return TPResult::Ambiguous();

      if (getLangOpts().CPlusPlus0x && isFollowedByBrace)
        return BracedCastResult;
      
      return TPResult::True();
    }
      
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
  case tok::annot_decltype:
    if (NextToken().is(tok::l_paren))
      return TPResult::Ambiguous();

    // This is a function-style cast in all cases we disambiguate other than
    // one:
    //   struct S {
    //     enum E : int { a = 4 }; // enum
    //     enum E : int { 4 };     // bit-field
    //   };
    if (getLangOpts().CPlusPlus0x && NextToken().is(tok::l_brace))
      return BracedCastResult;

    if (isStartOfObjCClassMessageMissingOpenBracket())
      return TPResult::False();
      
    return TPResult::True();

  // GNU typeof support.
  case tok::kw_typeof: {
    if (NextToken().isNot(tok::l_paren))
      return TPResult::True();

    TentativeParsingAction PA(*this);

    TPResult TPR = TryParseTypeofSpecifier();
    bool isFollowedByParen = Tok.is(tok::l_paren);
    bool isFollowedByBrace = Tok.is(tok::l_brace);

    PA.Revert();

    if (TPR == TPResult::Error())
      return TPResult::Error();

    if (isFollowedByParen)
      return TPResult::Ambiguous();

    if (getLangOpts().CPlusPlus0x && isFollowedByBrace)
      return BracedCastResult;

    return TPResult::True();
  }

  // C++0x type traits support
  case tok::kw___underlying_type:
    return TPResult::True();

  // C11 _Atomic
  case tok::kw__Atomic:
    return TPResult::True();

  default:
    return TPResult::False();
  }
}

/// [GNU] typeof-specifier:
///         'typeof' '(' expressions ')'
///         'typeof' '(' type-name ')'
///
Parser::TPResult Parser::TryParseTypeofSpecifier() {
  assert(Tok.is(tok::kw_typeof) && "Expected 'typeof'!");
  ConsumeToken();

  assert(Tok.is(tok::l_paren) && "Expected '('");
  // Parse through the parens after 'typeof'.
  ConsumeParen();
  if (!SkipUntil(tok::r_paren))
    return TPResult::Error();

  return TPResult::Ambiguous();
}

/// [ObjC] protocol-qualifiers:
////         '<' identifier-list '>'
Parser::TPResult Parser::TryParseProtocolQualifiers() {
  assert(Tok.is(tok::less) && "Expected '<' for qualifier list");
  ConsumeToken();
  do {
    if (Tok.isNot(tok::identifier))
      return TPResult::Error();
    ConsumeToken();
    
    if (Tok.is(tok::comma)) {
      ConsumeToken();
      continue;
    }
    
    if (Tok.is(tok::greater)) {
      ConsumeToken();
      return TPResult::Ambiguous();
    }
  } while (false);
  
  return TPResult::Error();
}

Parser::TPResult
Parser::TryParseDeclarationSpecifier(bool *HasMissingTypename) {
  TPResult TPR = isCXXDeclarationSpecifier(TPResult::False(),
                                           HasMissingTypename);
  if (TPR != TPResult::Ambiguous())
    return TPR;

  if (Tok.is(tok::kw_typeof))
    TryParseTypeofSpecifier();
  else {
    if (Tok.is(tok::annot_cxxscope))
      ConsumeToken();
    ConsumeToken();
    
    if (getLangOpts().ObjC1 && Tok.is(tok::less))
      TryParseProtocolQualifiers();
  }

  return TPResult::Ambiguous();
}

/// isCXXFunctionDeclarator - Disambiguates between a function declarator or
/// a constructor-style initializer, when parsing declaration statements.
/// Returns true for function declarator and false for constructor-style
/// initializer.
/// If during the disambiguation process a parsing error is encountered,
/// the function returns true to let the declaration parsing code handle it.
///
/// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///         exception-specification[opt]
///
bool Parser::isCXXFunctionDeclarator(bool *IsAmbiguous) {

  // C++ 8.2p1:
  // The ambiguity arising from the similarity between a function-style cast and
  // a declaration mentioned in 6.8 can also occur in the context of a
  // declaration. In that context, the choice is between a function declaration
  // with a redundant set of parentheses around a parameter name and an object
  // declaration with a function-style cast as the initializer. Just as for the
  // ambiguities mentioned in 6.8, the resolution is to consider any construct
  // that could possibly be a declaration a declaration.

  TentativeParsingAction PA(*this);

  ConsumeParen();
  bool InvalidAsDeclaration = false;
  TPResult TPR = TryParseParameterDeclarationClause(&InvalidAsDeclaration);
  if (TPR == TPResult::Ambiguous()) {
    if (Tok.isNot(tok::r_paren))
      TPR = TPResult::False();
    else {
      const Token &Next = NextToken();
      if (Next.is(tok::amp) || Next.is(tok::ampamp) ||
          Next.is(tok::kw_const) || Next.is(tok::kw_volatile) ||
          Next.is(tok::kw_throw) || Next.is(tok::kw_noexcept) ||
          Next.is(tok::l_square) || isCXX0XVirtSpecifier(Next) ||
          Next.is(tok::l_brace) || Next.is(tok::kw_try) ||
          Next.is(tok::equal) || Next.is(tok::arrow))
        // The next token cannot appear after a constructor-style initializer,
        // and can appear next in a function definition. This must be a function
        // declarator.
        TPR = TPResult::True();
      else if (InvalidAsDeclaration)
        // Use the absence of 'typename' as a tie-breaker.
        TPR = TPResult::False();
    }
  }

  PA.Revert();

  if (IsAmbiguous && TPR == TPResult::Ambiguous())
    *IsAmbiguous = true;

  // In case of an error, let the declaration parsing code handle it.
  return TPR != TPResult::False();
}

/// parameter-declaration-clause:
///   parameter-declaration-list[opt] '...'[opt]
///   parameter-declaration-list ',' '...'
///
/// parameter-declaration-list:
///   parameter-declaration
///   parameter-declaration-list ',' parameter-declaration
///
/// parameter-declaration:
///   attribute-specifier-seq[opt] decl-specifier-seq declarator attributes[opt]
///   attribute-specifier-seq[opt] decl-specifier-seq declarator attributes[opt]
///     '=' assignment-expression
///   attribute-specifier-seq[opt] decl-specifier-seq abstract-declarator[opt]
///     attributes[opt]
///   attribute-specifier-seq[opt] decl-specifier-seq abstract-declarator[opt]
///     attributes[opt] '=' assignment-expression
///
Parser::TPResult
Parser::TryParseParameterDeclarationClause(bool *InvalidAsDeclaration) {

  if (Tok.is(tok::r_paren))
    return TPResult::Ambiguous();

  //   parameter-declaration-list[opt] '...'[opt]
  //   parameter-declaration-list ',' '...'
  //
  // parameter-declaration-list:
  //   parameter-declaration
  //   parameter-declaration-list ',' parameter-declaration
  //
  while (1) {
    // '...'[opt]
    if (Tok.is(tok::ellipsis)) {
      ConsumeToken();
      if (Tok.is(tok::r_paren))
        return TPResult::True(); // '...)' is a sign of a function declarator.
      else
        return TPResult::False();
    }

    // An attribute-specifier-seq here is a sign of a function declarator.
    if (isCXX11AttributeSpecifier(/*Disambiguate*/false,
                                  /*OuterMightBeMessageSend*/true))
      return TPResult::True();

    ParsedAttributes attrs(AttrFactory);
    MaybeParseMicrosoftAttributes(attrs);

    // decl-specifier-seq
    // A parameter-declaration's initializer must be preceded by an '=', so
    // decl-specifier-seq '{' is not a parameter in C++11.
    TPResult TPR = TryParseDeclarationSpecifier(InvalidAsDeclaration);
    if (TPR != TPResult::Ambiguous())
      return TPR;

    // declarator
    // abstract-declarator[opt]
    TPR = TryParseDeclarator(true/*mayBeAbstract*/);
    if (TPR != TPResult::Ambiguous())
      return TPR;

    // [GNU] attributes[opt]
    if (Tok.is(tok::kw___attribute))
      return TPResult::True();

    if (Tok.is(tok::equal)) {
      // '=' assignment-expression
      // Parse through assignment-expression.
      if (!SkipUntil(tok::comma, tok::r_paren, true/*StopAtSemi*/,
                     true/*DontConsume*/))
        return TPResult::Error();
    }

    if (Tok.is(tok::ellipsis)) {
      ConsumeToken();
      if (Tok.is(tok::r_paren))
        return TPResult::True(); // '...)' is a sign of a function declarator.
      else
        return TPResult::False();
    }

    if (Tok.isNot(tok::comma))
      break;
    ConsumeToken(); // the comma.
  }

  return TPResult::Ambiguous();
}

/// TryParseFunctionDeclarator - We parsed a '(' and we want to try to continue
/// parsing as a function declarator.
/// If TryParseFunctionDeclarator fully parsed the function declarator, it will
/// return TPResult::Ambiguous(), otherwise it will return either False() or
/// Error().
///
/// '(' parameter-declaration-clause ')' cv-qualifier-seq[opt]
///         exception-specification[opt]
///
/// exception-specification:
///   'throw' '(' type-id-list[opt] ')'
///
Parser::TPResult Parser::TryParseFunctionDeclarator() {

  // The '(' is already parsed.

  TPResult TPR = TryParseParameterDeclarationClause();
  if (TPR == TPResult::Ambiguous() && Tok.isNot(tok::r_paren))
    TPR = TPResult::False();

  if (TPR == TPResult::False() || TPR == TPResult::Error())
    return TPR;

  // Parse through the parens.
  if (!SkipUntil(tok::r_paren))
    return TPResult::Error();

  // cv-qualifier-seq
  while (Tok.is(tok::kw_const)    ||
         Tok.is(tok::kw_volatile) ||
         Tok.is(tok::kw_restrict)   )
    ConsumeToken();

  // ref-qualifier[opt]
  if (Tok.is(tok::amp) || Tok.is(tok::ampamp))
    ConsumeToken();
  
  // exception-specification
  if (Tok.is(tok::kw_throw)) {
    ConsumeToken();
    if (Tok.isNot(tok::l_paren))
      return TPResult::Error();

    // Parse through the parens after 'throw'.
    ConsumeParen();
    if (!SkipUntil(tok::r_paren))
      return TPResult::Error();
  }
  if (Tok.is(tok::kw_noexcept)) {
    ConsumeToken();
    // Possibly an expression as well.
    if (Tok.is(tok::l_paren)) {
      // Find the matching rparen.
      ConsumeParen();
      if (!SkipUntil(tok::r_paren))
        return TPResult::Error();
    }
  }

  return TPResult::Ambiguous();
}

/// '[' constant-expression[opt] ']'
///
Parser::TPResult Parser::TryParseBracketDeclarator() {
  ConsumeBracket();
  if (!SkipUntil(tok::r_square))
    return TPResult::Error();

  return TPResult::Ambiguous();
}
