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

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIRECTIVETREE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIRECTIVETREE_H

#include "Token.h"
#include "clang/Basic/TokenKinds.h"
#include <optional>
#include <variant>
#include <vector>

namespace clang {
namespace clangd {

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
    /// std::nullopt indicates no branch was taken (e.g. #if 0 ... #endif).
    /// The initial tree from `parse()` has no branches marked as taken.
    /// See `chooseConditionalBranches()`.
    std::optional<unsigned> Taken;
  };

  /// Some piece of the file. {One of Code, Directive, Conditional}.
  using Chunk = std::variant<Code, Directive, Conditional>;
  std::vector<Chunk> Chunks;

  /// Extract preprocessor structure by examining the raw tokens.
  static DirectiveTree parse(const TokenStream &);

  /// Produce a parseable token stream by stripping all directive tokens.
  ///
  /// Conditional sections are replaced by the taken branch, if any.
  /// This tree must describe the provided token stream.
  TokenStream stripDirectives(const TokenStream &) const;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const DirectiveTree &);
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

/// Pairs preprocessor conditional directives and computes their token ranges.
std::vector<Token::Range> pairDirectiveRanges(const DirectiveTree &Tree,
                                              const TokenStream &Code);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_DIRECTIVETREE_H
