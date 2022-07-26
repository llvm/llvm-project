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
  assert(Declarator->symbol() == (SymbolID)(cxx::Symbol::declarator));
  bool IsFunction = false;
  using cxx::Rule;
  while (true) {
    // not well-formed code, return the best guess.
    if (Declarator->kind() != ForestNode::Sequence)
      return IsFunction;

    switch ((cxx::Rule)Declarator->rule()) {
    case Rule::noptr_declarator_0declarator_id: // reached the bottom
      return IsFunction;
    // *X is a nonfunction (unless X is a function).
    case Rule::ptr_declarator_0ptr_operator_1ptr_declarator:
      Declarator = Declarator->elements()[1];
      IsFunction = false;
      continue;
    // X() is a function (unless X is a pointer or similar).
    case Rule::
        declarator_0noptr_declarator_1parameters_and_qualifiers_2trailing_return_type:
    case Rule::noptr_declarator_0noptr_declarator_1parameters_and_qualifiers:
      Declarator = Declarator->elements()[0];
      IsFunction = true;
      continue;
    // X[] is an array (unless X is a pointer or function).
    case Rule::
        noptr_declarator_0noptr_declarator_1l_square_2constant_expression_3r_square:
    case Rule::noptr_declarator_0noptr_declarator_1l_square_2r_square:
      Declarator = Declarator->elements()[0];
      IsFunction = false;
      continue;
    // (X) is whatever X is.
    case Rule::noptr_declarator_0l_paren_1ptr_declarator_2r_paren:
      Declarator = Declarator->elements()[1];
      continue;
    case Rule::ptr_declarator_0noptr_declarator:
    case Rule::declarator_0ptr_declarator:
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
    assert(N->symbol() == (SymbolID)Symbol::decl_specifier_seq ||
           N->symbol() == (SymbolID)Symbol::type_specifier_seq ||
           N->symbol() == (SymbolID)Symbol::defining_type_specifier_seq ||
           N->symbol() == (SymbolID)Symbol::decl_specifier ||
           N->symbol() == (SymbolID)Symbol::type_specifier ||
           N->symbol() == (SymbolID)Symbol::defining_type_specifier ||
           N->symbol() == (SymbolID)Symbol::simple_type_specifier);
    if (N->kind() == ForestNode::Opaque)
      return false; // conservative
    if (N->kind() == ForestNode::Ambiguous)
      return llvm::all_of(N->alternatives(), hasExclusiveType); // conservative
    // All supported symbols are nonterminals.
    assert(N->kind() == ForestNode::Sequence);
    switch (N->rule()) {
      // seq := element seq: check element then continue into seq
      case (RuleID)Rule::decl_specifier_seq_0decl_specifier_1decl_specifier_seq:
      case (RuleID)Rule::defining_type_specifier_seq_0defining_type_specifier_1defining_type_specifier_seq:
      case (RuleID)Rule::type_specifier_seq_0type_specifier_1type_specifier_seq:
        if (hasExclusiveType(N->children()[0]))
          return true;
        N = N->children()[1];
        continue;
      // seq := element: continue into element
      case (RuleID)Rule::decl_specifier_seq_0decl_specifier:
      case (RuleID)Rule::type_specifier_seq_0type_specifier:
      case (RuleID)Rule::defining_type_specifier_seq_0defining_type_specifier:
        N = N->children()[0];
        continue;

      // defining-type-specifier
      case (RuleID)Rule::defining_type_specifier_0type_specifier:
        N = N->children()[0];
        continue;
      case (RuleID)Rule::defining_type_specifier_0class_specifier:
      case (RuleID)Rule::defining_type_specifier_0enum_specifier:
        return true;

      // decl-specifier
      case (RuleID)Rule::decl_specifier_0defining_type_specifier:
        N = N->children()[0];
        continue;
      case (RuleID)Rule::decl_specifier_0consteval:
      case (RuleID)Rule::decl_specifier_0constexpr:
      case (RuleID)Rule::decl_specifier_0constinit:
      case (RuleID)Rule::decl_specifier_0inline:
      case (RuleID)Rule::decl_specifier_0friend:
      case (RuleID)Rule::decl_specifier_0storage_class_specifier:
      case (RuleID)Rule::decl_specifier_0typedef:
      case (RuleID)Rule::decl_specifier_0function_specifier:
        return false;

      // type-specifier
      case (RuleID)Rule::type_specifier_0elaborated_type_specifier:
      case (RuleID)Rule::type_specifier_0typename_specifier:
        return true;
      case (RuleID)Rule::type_specifier_0simple_type_specifier:
        N = N->children()[0];
        continue;
      case (RuleID)Rule::type_specifier_0cv_qualifier:
        return false;

      // simple-type-specifier
      case (RuleID)Rule::simple_type_specifier_0type_name:
      case (RuleID)Rule::simple_type_specifier_0template_name:
      case (RuleID)Rule::simple_type_specifier_0builtin_type:
      case (RuleID)Rule::simple_type_specifier_0nested_name_specifier_1template_2simple_template_id:
      case (RuleID)Rule::simple_type_specifier_0nested_name_specifier_1template_name:
      case (RuleID)Rule::simple_type_specifier_0nested_name_specifier_1type_name:
      case (RuleID)Rule::simple_type_specifier_0decltype_specifier:
      case (RuleID)Rule::simple_type_specifier_0placeholder_type_specifier:
        return true;
      case (RuleID)Rule::simple_type_specifier_0long:
      case (RuleID)Rule::simple_type_specifier_0short:
      case (RuleID)Rule::simple_type_specifier_0signed:
      case (RuleID)Rule::simple_type_specifier_0unsigned:
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
    const ForestNode &N = onlySymbol((SymbolID)Symbol::kind, P.RHS, P.Tokens); \
    return cond;                                                               \
  }
  return {
      {(RuleID)Rule::function_declarator_0declarator,
       SYMBOL_GUARD(declarator, isFunctionDeclarator(&N))},
      {(RuleID)Rule::non_function_declarator_0declarator,
       SYMBOL_GUARD(declarator, !isFunctionDeclarator(&N))},

      // A {decl,type,defining-type}-specifier-sequence cannot have multiple
      // "exclusive" types (like class names): a value has only one type.
      {(RuleID)Rule::
           defining_type_specifier_seq_0defining_type_specifier_1defining_type_specifier_seq,
       GUARD(!hasExclusiveType(P.RHS[0]) || !hasExclusiveType(P.RHS[1]))},
      {(RuleID)Rule::type_specifier_seq_0type_specifier_1type_specifier_seq,
       GUARD(!hasExclusiveType(P.RHS[0]) || !hasExclusiveType(P.RHS[1]))},
      {(RuleID)Rule::decl_specifier_seq_0decl_specifier_1decl_specifier_seq,
       GUARD(!hasExclusiveType(P.RHS[0]) || !hasExclusiveType(P.RHS[1]))},

      {(RuleID)Rule::contextual_override_0identifier,
       TOKEN_GUARD(identifier, Tok.text() == "override")},
      {(RuleID)Rule::contextual_final_0identifier,
       TOKEN_GUARD(identifier, Tok.text() == "final")},
      {(RuleID)Rule::import_keyword_0identifier,
       TOKEN_GUARD(identifier, Tok.text() == "import")},
      {(RuleID)Rule::export_keyword_0identifier,
       TOKEN_GUARD(identifier, Tok.text() == "export")},
      {(RuleID)Rule::module_keyword_0identifier,
       TOKEN_GUARD(identifier, Tok.text() == "module")},
      {(RuleID)Rule::contextual_zero_0numeric_constant,
       TOKEN_GUARD(numeric_constant, Tok.text() == "0")},

      {(RuleID)Rule::
           selection_statement_0if_1l_paren_2condition_3r_paren_4statement,
       guardNextTokenNotElse},
      {(RuleID)Rule::
           selection_statement_0if_1constexpr_2l_paren_3condition_4r_paren_5statement,
       guardNextTokenNotElse},

      // The grammar distinguishes (only) user-defined vs plain string literals,
      // where the clang lexer distinguishes (only) encoding types.
      {(RuleID)Rule::user_defined_string_literal_chunk_0string_literal,
       TOKEN_GUARD(string_literal, isStringUserDefined(Tok))},
      {(RuleID)Rule::user_defined_string_literal_chunk_0utf8_string_literal,
       TOKEN_GUARD(utf8_string_literal, isStringUserDefined(Tok))},
      {(RuleID)Rule::user_defined_string_literal_chunk_0utf16_string_literal,
       TOKEN_GUARD(utf16_string_literal, isStringUserDefined(Tok))},
      {(RuleID)Rule::user_defined_string_literal_chunk_0utf32_string_literal,
       TOKEN_GUARD(utf32_string_literal, isStringUserDefined(Tok))},
      {(RuleID)Rule::user_defined_string_literal_chunk_0wide_string_literal,
       TOKEN_GUARD(wide_string_literal, isStringUserDefined(Tok))},
      {(RuleID)Rule::string_literal_chunk_0string_literal,
       TOKEN_GUARD(string_literal, !isStringUserDefined(Tok))},
      {(RuleID)Rule::string_literal_chunk_0utf8_string_literal,
       TOKEN_GUARD(utf8_string_literal, !isStringUserDefined(Tok))},
      {(RuleID)Rule::string_literal_chunk_0utf16_string_literal,
       TOKEN_GUARD(utf16_string_literal, !isStringUserDefined(Tok))},
      {(RuleID)Rule::string_literal_chunk_0utf32_string_literal,
       TOKEN_GUARD(utf32_string_literal, !isStringUserDefined(Tok))},
      {(RuleID)Rule::string_literal_chunk_0wide_string_literal,
       TOKEN_GUARD(wide_string_literal, !isStringUserDefined(Tok))},
      // And the same for chars.
      {(RuleID)Rule::user_defined_character_literal_0char_constant,
       TOKEN_GUARD(char_constant, isCharUserDefined(Tok))},
      {(RuleID)Rule::user_defined_character_literal_0utf8_char_constant,
       TOKEN_GUARD(utf8_char_constant, isCharUserDefined(Tok))},
      {(RuleID)Rule::user_defined_character_literal_0utf16_char_constant,
       TOKEN_GUARD(utf16_char_constant, isCharUserDefined(Tok))},
      {(RuleID)Rule::user_defined_character_literal_0utf32_char_constant,
       TOKEN_GUARD(utf32_char_constant, isCharUserDefined(Tok))},
      {(RuleID)Rule::user_defined_character_literal_0wide_char_constant,
       TOKEN_GUARD(wide_char_constant, isCharUserDefined(Tok))},
      {(RuleID)Rule::character_literal_0char_constant,
       TOKEN_GUARD(char_constant, !isCharUserDefined(Tok))},
      {(RuleID)Rule::character_literal_0utf8_char_constant,
       TOKEN_GUARD(utf8_char_constant, !isCharUserDefined(Tok))},
      {(RuleID)Rule::character_literal_0utf16_char_constant,
       TOKEN_GUARD(utf16_char_constant, !isCharUserDefined(Tok))},
      {(RuleID)Rule::character_literal_0utf32_char_constant,
       TOKEN_GUARD(utf32_char_constant, !isCharUserDefined(Tok))},
      {(RuleID)Rule::character_literal_0wide_char_constant,
       TOKEN_GUARD(wide_char_constant, !isCharUserDefined(Tok))},
      // clang just has one NUMERIC_CONSTANT token for {ud,plain}x{float,int}
      {(RuleID)Rule::user_defined_integer_literal_0numeric_constant,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == (Integer | UserDefined))},
      {(RuleID)Rule::user_defined_floating_point_literal_0numeric_constant,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == (Floating | UserDefined))},
      {(RuleID)Rule::integer_literal_0numeric_constant,
       TOKEN_GUARD(numeric_constant, numKind(Tok) == Integer)},
      {(RuleID)Rule::floating_point_literal_0numeric_constant,
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
      {(ExtensionID)Extension::Brackets, recoverBrackets},
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
