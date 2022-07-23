//===--- CXX.cpp - Define public interfaces for C++ grammar ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/cxx/CXX.h"
#include "clang-pseudo/Forest.h"
#include "clang-pseudo/Language.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include <utility>
#define DEBUG_TYPE "CXX.cpp"

namespace clang {
namespace pseudo {
namespace cxx {
namespace {
static const char *CXXBNF =
#include "CXXBNF.inc"
    ;

// User-defined string literals look like `""suffix`.
bool isStringUserDefined(const Token &Tok) {
  return !Tok.text().endswith("\"");
}
bool isCharUserDefined(const Token &Tok) { return !Tok.text().endswith("'"); }

// Combinable flags describing numbers.
// Clang has just one numeric_token kind, the grammar has 4.
enum NumericKind {
  Integer = 0,
  Floating = 1 << 0,
  UserDefined = 1 << 1,
};
// Determine the kind of numeric_constant we have.
// We can assume it's something valid, as it has been lexed.
// FIXME: is this expensive enough that we should set flags on the token
// and reuse them rather than computing it for each guard?
unsigned numKind(const Token &Tok) {
  assert(Tok.Kind == tok::numeric_constant);
  llvm::StringRef Text = Tok.text();
  if (Text.size() <= 1)
    return Integer;
  bool Hex =
      Text.size() > 2 && Text[0] == '0' && (Text[1] == 'x' || Text[1] == 'X');
  uint8_t K = Integer;

  for (char C : Text) {
    switch (C) {
    case '.':
      K |= Floating;
      break;
    case 'e':
    case 'E':
      if (!Hex)
        K |= Floating;
      break;
    case 'p':
    case 'P':
      if (Hex)
        K |= Floating;
      break;
    case '_':
      K |= UserDefined;
      break;
    default:
      break;
    }
  }

  // We would be done here, but there are stdlib UDLs that lack _.
  // We must distinguish these from the builtin suffixes.
  unsigned LastLetter = Text.size();
  while (LastLetter > 0 && isLetter(Text[LastLetter - 1]))
    --LastLetter;
  if (LastLetter == Text.size()) // Common case
    return NumericKind(K);
  // Trailing d/e/f are not part of the suffix in hex numbers.
  while (Hex && LastLetter < Text.size() && isHexDigit(Text[LastLetter]))
    ++LastLetter;
  return llvm::StringSwitch<int, unsigned>(Text.substr(LastLetter))
      // std::chrono
      .Cases("h", "min", "s", "ms", "us", "ns", "d", "y", K | UserDefined)
      // complex
      .Cases("il", "i", "if", K | UserDefined)
      .Default(K);
}

// RHS is expected to contain a single terminal.
// Returns the corresponding token.
const Token &onlyToken(tok::TokenKind Kind,
                       const ArrayRef<const ForestNode *> RHS,
                       const TokenStream &Tokens) {
  assert(RHS.size() == 1 && RHS.front()->symbol() == tokenSymbol(Kind));
  return Tokens.tokens()[RHS.front()->startTokenIndex()];
}
// RHS is expected to contain a single symbol.
// Returns the corresponding ForestNode.
const ForestNode &onlySymbol(SymbolID Kind,
                             const ArrayRef<const ForestNode *> RHS,
                             const TokenStream &Tokens) {
  assert(RHS.size() == 1 && RHS.front()->symbol() == Kind);
  return *RHS.front();
}

bool isFunctionDeclarator(const ForestNode *Declarator) {
  assert(Declarator->symbol() == cxx::Symbol::declarator);
  bool IsFunction = false;
  while (true) {
    // not well-formed code, return the best guess.
    if (Declarator->kind() != ForestNode::Sequence)
      return IsFunction;

    switch (Declarator->rule()) {
    case rule::noptr_declarator::declarator_id: // reached the bottom
      return IsFunction;
    // *X is a nonfunction (unless X is a function).
    case rule::ptr_declarator::ptr_operator__ptr_declarator:
      Declarator = Declarator->elements()[1];
      IsFunction = false;
      continue;
    // X() is a function (unless X is a pointer or similar).
    case rule::declarator::
        noptr_declarator__parameters_and_qualifiers__trailing_return_type:
    case rule::noptr_declarator::noptr_declarator__parameters_and_qualifiers:
      Declarator = Declarator->elements()[0];
      IsFunction = true;
      continue;
    // X[] is an array (unless X is a pointer or function).
    case rule::noptr_declarator::
        noptr_declarator__L_SQUARE__constant_expression__R_SQUARE:
    case rule::noptr_declarator::noptr_declarator__L_SQUARE__R_SQUARE:
      Declarator = Declarator->elements()[0];
      IsFunction = false;
      continue;
    // (X) is whatever X is.
    case rule::noptr_declarator::L_PAREN__ptr_declarator__R_PAREN:
      Declarator = Declarator->elements()[1];
      continue;
    case rule::ptr_declarator::noptr_declarator:
    case rule::declarator::ptr_declarator:
      Declarator = Declarator->elements()[0];
      continue;

    default:
      assert(false && "unhandled declarator for IsFunction");
      return IsFunction;
    }
  }
  llvm_unreachable("unreachable");
}

bool guardNextTokenNotElse(const GuardParams &P) {
  return symbolToToken(P.Lookahead) != tok::kw_else;
}

// Whether this e.g. decl-specifier contains an "exclusive" type such as a class
// name, and thus can't combine with a second exclusive type.
//
// Returns false for
//  - non-types
//  - "unsigned" etc that may suffice as types but may modify others
//  - cases of uncertainty (e.g. due to ambiguity)
bool hasExclusiveType(const ForestNode *N) {
  // FIXME: every time we apply this check, we walk the whole subtree.
  // Add per-node caching instead.
  while (true) {
    assert(N->symbol() == Symbol::decl_specifier_seq ||
           N->symbol() == Symbol::type_specifier_seq ||
           N->symbol() == Symbol::defining_type_specifier_seq ||
           N->symbol() == Symbol::decl_specifier ||
           N->symbol() == Symbol::type_specifier ||
           N->symbol() == Symbol::defining_type_specifier ||
           N->symbol() == Symbol::simple_type_specifier);
    if (N->kind() == ForestNode::Opaque)
      return false; // conservative
    if (N->kind() == ForestNode::Ambiguous)
      return llvm::all_of(N->alternatives(), hasExclusiveType); // conservative
    // All supported symbols are nonterminals.
    assert(N->kind() == ForestNode::Sequence);
    switch (N->rule()) {
      // seq := element seq: check element then continue into seq
      case rule::decl_specifier_seq::decl_specifier__decl_specifier_seq:
      case rule::defining_type_specifier_seq::defining_type_specifier__defining_type_specifier_seq:
      case rule::type_specifier_seq::type_specifier__type_specifier_seq:
        if (hasExclusiveType(N->children()[0]))
          return true;
        N = N->children()[1];
        continue;
      // seq := element: continue into element
      case rule::decl_specifier_seq::decl_specifier:
      case rule::type_specifier_seq::type_specifier:
      case rule::defining_type_specifier_seq::defining_type_specifier:
        N = N->children()[0];
        continue;

      // defining-type-specifier
      case rule::defining_type_specifier::type_specifier:
        N = N->children()[0];
        continue;
      case rule::defining_type_specifier::class_specifier:
      case rule::defining_type_specifier::enum_specifier:
        return true;

      // decl-specifier
      case rule::decl_specifier::defining_type_specifier:
        N = N->children()[0];
        continue;
      case rule::decl_specifier::CONSTEVAL:
      case rule::decl_specifier::CONSTEXPR:
      case rule::decl_specifier::CONSTINIT:
      case rule::decl_specifier::INLINE:
      case rule::decl_specifier::FRIEND:
      case rule::decl_specifier::storage_class_specifier:
      case rule::decl_specifier::TYPEDEF:
      case rule::decl_specifier::function_specifier:
        return false;

      // type-specifier
      case rule::type_specifier::elaborated_type_specifier:
      case rule::type_specifier::typename_specifier:
        return true;
      case rule::type_specifier::simple_type_specifier:
        N = N->children()[0];
        continue;
      case rule::type_specifier::cv_qualifier:
        return false;

      // simple-type-specifier
      case rule::simple_type_specifier::type_name:
      case rule::simple_type_specifier::template_name:
      case rule::simple_type_specifier::builtin_type:
      case rule::simple_type_specifier::nested_name_specifier__TEMPLATE__simple_template_id:
      case rule::simple_type_specifier::nested_name_specifier__template_name:
      case rule::simple_type_specifier::nested_name_specifier__type_name:
      case rule::simple_type_specifier::decltype_specifier:
      case rule::simple_type_specifier::placeholder_type_specifier:
        return true;
      case rule::simple_type_specifier::LONG:
      case rule::simple_type_specifier::SHORT:
      case rule::simple_type_specifier::SIGNED:
      case rule::simple_type_specifier::UNSIGNED:
        return false;

      default:
        LLVM_DEBUG(llvm::errs() << "Unhandled rule " << N->rule() << "\n");
        llvm_unreachable("hasExclusiveType be exhaustive!");
    }
  }
}

llvm::DenseMap<ExtensionID, RuleGuard> buildGuards() {
#define GUARD(cond)                                                            \
  {                                                                            \
    [](const GuardParams &P) { return cond; }                                  \
  }
#define TOKEN_GUARD(kind, cond)                                                \
  [](const GuardParams& P) {                                                   \
    const Token &Tok = onlyToken(tok::kind, P.RHS, P.Tokens);                  \
    return cond;                                                               \
  }
#define SYMBOL_GUARD(kind, cond)                                               \
  [](const GuardParams& P) {                                                   \
    const ForestNode &N = onlySymbol(Symbol::kind, P.RHS, P.Tokens); \
    return cond;                                                               \
  }
  return {
      {rule::function_declarator::declarator,
       SYMBOL_GUARD(declarator, isFunctionDeclarator(&N))},
      {rule::non_function_declarator::declarator,
       SYMBOL_GUARD(declarator, !isFunctionDeclarator(&N))},

      // A {decl,type,defining-type}-specifier-sequence cannot have multiple
      // "exclusive" types (like class names): a value has only one type.
      {rule::defining_type_specifier_seq::
           defining_type_specifier__defining_type_specifier_seq,
       GUARD(!hasExclusiveType(P.RHS[0]) || !hasExclusiveType(P.RHS[1]))},
      {rule::type_specifier_seq::type_specifier__type_specifier_seq,
       GUARD(!hasExclusiveType(P.RHS[0]) || !hasExclusiveType(P.RHS[1]))},
      {rule::decl_specifier_seq::decl_specifier__decl_specifier_seq,
       GUARD(!hasExclusiveType(P.RHS[0]) || !hasExclusiveType(P.RHS[1]))},

      {rule::contextual_override::IDENTIFIER,
       TOKEN_GUARD(identifier, Tok.text() == "override")},
      {rule::contextual_final::IDENTIFIER,
       TOKEN_GUARD(identifier, Tok.text() == "final")},
      {rule::import_keyword::IDENTIFIER,
       TOKEN_GUARD(identifier, Tok.text() == "import")},
      {rule::export_keyword::IDENTIFIER,
       TOKEN_GUARD(identifier, Tok.text() == "export")},
      {rule::module_keyword::IDENTIFIER,
       TOKEN_GUARD(identifier, Tok.text() == "module")},
      {rule::contextual_zero::NUMERIC_CONSTANT,
       TOKEN_GUARD(numeric_constant, Tok.text() == "0")},

      // FIXME: the init-statement variants are missing?
      {rule::selection_statement::IF__L_PAREN__condition__R_PAREN__statement,
       guardNextTokenNotElse},
      {rule::selection_statement::
           IF__CONSTEXPR__L_PAREN__condition__R_PAREN__statement,
       guardNextTokenNotElse},

      // The grammar distinguishes (only) user-defined vs plain string literals,
      // where the clang lexer distinguishes (only) encoding types.
      {rule::user_defined_string_literal_chunk::STRING_LITERAL,
       TOKEN_GUARD(string_literal, isStringUserDefined(Tok))},
      {rule::user_defined_string_literal_chunk::UTF8_STRING_LITERAL,
       TOKEN_GUARD(utf8_string_literal, isStringUserDefined(Tok))},
      {rule::user_defined_string_literal_chunk::UTF16_STRING_LITERAL,
       TOKEN_GUARD(utf16_string_literal, isStringUserDefined(Tok))},
      {rule::user_defined_string_literal_chunk::UTF32_STRING_LITERAL,
       TOKEN_GUARD(utf32_string_literal, isStringUserDefined(Tok))},
      {rule::user_defined_string_literal_chunk::WIDE_STRING_LITERAL,
       TOKEN_GUARD(wide_string_literal, isStringUserDefined(Tok))},
      {rule::string_literal_chunk::STRING_LITERAL,
       TOKEN_GUARD(string_literal, !isStringUserDefined(Tok))},
      {rule::string_literal_chunk::UTF8_STRING_LITERAL,
       TOKEN_GUARD(utf8_string_literal, !isStringUserDefined(Tok))},
      {rule::string_literal_chunk::UTF16_STRING_LITERAL,
       TOKEN_GUARD(utf16_string_literal, !isStringUserDefined(Tok))},
      {rule::string_literal_chunk::UTF32_STRING_LITERAL,
       TOKEN_GUARD(utf32_string_literal, !isStringUserDefined(Tok))},
      {rule::string_literal_chunk::WIDE_STRING_LITERAL,
       TOKEN_GUARD(wide_string_literal, !isStringUserDefined(Tok))},
      // And the same for chars.
      {rule::user_defined_character_literal::CHAR_CONSTANT,
       TOKEN_GUARD(char_constant, isCharUserDefined(Tok))},
      {rule::user_defined_character_literal::UTF8_CHAR_CONSTANT,
       TOKEN_GUARD(utf8_char_constant, isCharUserDefined(Tok))},
      {rule::user_defined_character_literal::UTF16_CHAR_CONSTANT,
       TOKEN_GUARD(utf16_char_constant, isCharUserDefined(Tok))},
      {rule::user_defined_character_literal::UTF32_CHAR_CONSTANT,
       TOKEN_GUARD(utf32_char_constant, isCharUserDefined(Tok))},
      {rule::user_defined_character_literal::WIDE_CHAR_CONSTANT,
       TOKEN_GUARD(wide_char_constant, isCharUserDefined(Tok))},
      {rule::character_literal::CHAR_CONSTANT,
       TOKEN_GUARD(char_constant, !isCharUserDefined(Tok))},
      {rule::character_literal::UTF8_CHAR_CONSTANT,
       TOKEN_GUARD(utf8_char_constant, !isCharUserDefined(Tok))},
      {rule::character_literal::UTF16_CHAR_CONSTANT,
       TOKEN_GUARD(utf16_char_constant, !isCharUserDefined(Tok))},
      {rule::character_literal::UTF32_CHAR_CONSTANT,
       TOKEN_GUARD(utf32_char_constant, !isCharUserDefined(Tok))},
      {rule::character_literal::WIDE_CHAR_CONSTANT,
       TOKEN_GUARD(wide_char_constant, !isCharUserDefined(Tok))},
      // clang just has one NUMERIC_CONSTANT token for {ud,plain}x{float,int}
      {rule::user_defined_integer_literal::NUMERIC_CONSTANT,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == (Integer | UserDefined))},
      {rule::user_defined_floating_point_literal::NUMERIC_CONSTANT,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == (Floating | UserDefined))},
      {rule::integer_literal::NUMERIC_CONSTANT,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == Integer)},
      {rule::floating_point_literal::NUMERIC_CONSTANT,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == Floating)},
  };
#undef TOKEN_GUARD
#undef SYMBOL_GUARD
}

Token::Index recoverBrackets(Token::Index Begin, const TokenStream &Tokens) {
  assert(Begin > 0);
  const Token &Left = Tokens.tokens()[Begin - 1];
  assert(Left.Kind == tok::l_brace || Left.Kind == tok::l_paren ||
         Left.Kind == tok::l_square);
  if (const Token *Right = Left.pair()) {
    assert(Tokens.index(*Right) > Begin - 1);
    return Tokens.index(*Right);
  }
  return Token::Invalid;
}

llvm::DenseMap<ExtensionID, RecoveryStrategy> buildRecoveryStrategies() {
  return {
      {Extension::Brackets, recoverBrackets},
  };
}

} // namespace

const Language &getLanguage() {
  static const auto &CXXLanguage = []() -> const Language & {
    std::vector<std::string> Diags;
    auto G = Grammar::parseBNF(CXXBNF, Diags);
    assert(Diags.empty());
    LRTable Table = LRTable::buildSLR(G);
    const Language *PL = new Language{
        std::move(G),
        std::move(Table),
        buildGuards(),
        buildRecoveryStrategies(),
    };
    return *PL;
  }();
  return CXXLanguage;
}

} // namespace cxx
} // namespace pseudo
} // namespace clang
