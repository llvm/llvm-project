//===--- DirectiveTree.h - Find and strip preprocessor directives *- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pseudoparser tries to match a token stream to the C++ grammar.
// Preprocessor #defines and other directives are not part of this grammar, and
// should be removed before the file can be parsed.
//
// Conditional blocks like #if...#else...#endif are particularly tricky, as
// simply stripping the directives may not produce a grammatical result:
//
//   return
//     #ifndef DEBUG
//       1
//     #else
//       0
//     #endif
//       ;
//
// This header supports analyzing and removing the directives in a source file.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_DIRECTIVETREE_H
#define CLANG_PSEUDO_DIRECTIVETREE_H

#include "clang-pseudo/Token.h"
#include "clang/Basic/TokenKinds.h"
#include <vector>

namespace clang {
class LangOptions;
namespace pseudo {

/// Describes the structure of a source file, as seen by the preprocessor.
///
/// The structure is a tree, whose leaves are plain source code and directives,
/// and whose internal nodes are #if...#endif sections.
///
/// (root)
/// |-+ Directive                    #include <stdio.h>
/// |-+ Code                         int main() {
/// | `                                printf("hello, ");
/// |-+ Conditional -+ Directive     #ifndef NDEBUG
/// | |-+ Code                         printf("debug\n");
/// | |-+ Directive                  #else
/// | |-+ Code                         printf("production\n");
/// | `-+ Directive                  #endif
/// |-+ Code                           return 0;
///   `                              }
///
/// Unlike the clang preprocessor, we model the full tree explicitly.
/// This class does not recognize macro usage, only directives.
struct DirectiveTree {
  /// A range of code (and possibly comments) containing no directives.
  struct Code {
    Token::Range Tokens;
  };
  /// A preprocessor directive.
  struct Directive {
    /// Raw tokens making up the directive, starting with `#`.
    Token::Range Tokens;
    clang::tok::PPKeywordKind Kind = clang::tok::pp_not_keyword;
  };
  /// A preprocessor conditional section.
  ///
  /// This starts with an #if, #ifdef, #ifndef etc directive.
  /// It covers all #else branches, and spans until the matching #endif.
  struct Conditional {
    /// The sequence of directives that introduce top-level alternative parses.
    ///
    /// The first branch will have an #if type directive.
    /// Subsequent branches will have #else type directives.
    std::vector<std::pair<Directive, DirectiveTree>> Branches;
    /// The directive terminating the conditional, should be #endif.
    Directive End;
    /// The index of the conditional branch we chose as active.
    /// None indicates no branch was taken (e.g. #if 0 ... #endif).
    /// The initial tree from `parse()` has no branches marked as taken.
    /// See `chooseConditionalBranches()`.
    llvm::Optional<unsigned> Taken;
  };

  /// Some piece of the file. {One of Code, Directive, Conditional}.
  class Chunk; // Defined below.
  std::vector<Chunk> Chunks;

  /// Extract preprocessor structure by examining the raw tokens.
  static DirectiveTree parse(const TokenStream &);

  // FIXME: allow deriving a preprocessed stream
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const DirectiveTree &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const DirectiveTree::Chunk &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const DirectiveTree::Code &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                              const DirectiveTree::Directive &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                              const DirectiveTree::Conditional &);

/// Selects a "taken" branch for each conditional directive in the file.
///
/// The choice is somewhat arbitrary, but aims to produce a useful parse:
///  - idioms like `#if 0` are respected
///  - we avoid paths that reach `#error`
///  - we try to maximize the amount of code seen
/// The choice may also be "no branch taken".
///
/// Choices are also made for conditionals themselves inside not-taken branches:
///   #if 1 // taken!
///   #else // not taken
///      #if 1 // taken!
///      #endif
///   #endif
///
/// The choices are stored in Conditional::Taken nodes.
void chooseConditionalBranches(DirectiveTree &, const TokenStream &Code);

// FIXME: This approximates std::variant<Code, Directive, Conditional>.
//         Switch once we can use C++17.
class DirectiveTree::Chunk {
public:
  enum Kind { K_Empty, K_Code, K_Directive, K_Conditional };
  Kind kind() const {
    return CodeVariant          ? K_Code
           : DirectiveVariant   ? K_Directive
           : ConditionalVariant ? K_Conditional
                                : K_Empty;
  }

  Chunk() = delete;
  Chunk(const Chunk &) = delete;
  Chunk(Chunk &&) = default;
  Chunk &operator=(const Chunk &) = delete;
  Chunk &operator=(Chunk &&) = default;
  ~Chunk() = default;

  // T => Chunk constructor.
  Chunk(Code C) : CodeVariant(std::move(C)) {}
  Chunk(Directive C) : DirectiveVariant(std::move(C)) {}
  Chunk(Conditional C) : ConditionalVariant(std::move(C)) {}

  // Chunk => T& and const T& conversions.
#define CONVERSION(CONST, V)                                                   \
  explicit operator CONST V &() CONST { return *V##Variant; }
  CONVERSION(const, Code);
  CONVERSION(, Code);
  CONVERSION(const, Directive);
  CONVERSION(, Directive);
  CONVERSION(const, Conditional);
  CONVERSION(, Conditional);
#undef CONVERSION

private:
  // Wasteful, a union variant would be better!
  llvm::Optional<Code> CodeVariant;
  llvm::Optional<Directive> DirectiveVariant;
  llvm::Optional<Conditional> ConditionalVariant;
};

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_DIRECTIVETREE_H
