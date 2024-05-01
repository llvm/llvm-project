//===-- lib/Parser/preprocessor.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_PREPROCESSOR_H_
#define FORTRAN_PARSER_PREPROCESSOR_H_

// A Fortran-aware preprocessing module used by the prescanner to implement
// preprocessing directives and macro replacement.  Intended to be efficient
// enough to always run on all source files even when no preprocessing is
// performed, so that special compiler command options &/or source file name
// extensions for preprocessing will not be necessary.

#include "flang/Parser/char-block.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/token-sequence.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <list>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace Fortran::parser {

class Prescanner;
class Preprocessor;

// Defines a macro
class Definition {
public:
  Definition(const TokenSequence &, std::size_t firstToken, std::size_t tokens);
  Definition(const std::vector<std::string> &argNames, const TokenSequence &,
      std::size_t firstToken, std::size_t tokens, bool isVariadic = false);
  Definition(const std::string &predefined, AllSources &);

  bool isFunctionLike() const { return isFunctionLike_; }
  std::size_t argumentCount() const { return argNames_.size(); }
  bool isVariadic() const { return isVariadic_; }
  bool isDisabled() const { return isDisabled_; }
  bool isPredefined() const { return isPredefined_; }
  const TokenSequence &replacement() const { return replacement_; }

  bool set_isDisabled(bool disable);

  TokenSequence Apply(const std::vector<TokenSequence> &args, Prescanner &);

  void Print(llvm::raw_ostream &out, const char *macroName = "") const;

private:
  static TokenSequence Tokenize(const std::vector<std::string> &argNames,
      const TokenSequence &token, std::size_t firstToken, std::size_t tokens);
  // For a given token, return the index of the argument to which the token
  // corresponds, or `argumentCount` if the token does not correspond to any
  // argument.
  std::size_t GetArgumentIndex(const CharBlock &token) const;

  bool isFunctionLike_{false};
  bool isVariadic_{false};
  bool isDisabled_{false};
  bool isPredefined_{false};
  std::vector<std::string> argNames_;
  TokenSequence replacement_;
};

// Preprocessing state
class Preprocessor {
public:
  explicit Preprocessor(AllSources &);

  const AllSources &allSources() const { return allSources_; }
  AllSources &allSources() { return allSources_; }

  void DefineStandardMacros();
  void Define(const std::string &macro, const std::string &value);
  void Undefine(std::string macro);
  bool IsNameDefined(const CharBlock &);
  bool IsFunctionLikeDefinition(const CharBlock &);
  bool AnyDefinitions() const { return !definitions_.empty(); }

  // When called with partialFunctionLikeMacro not null, MacroReplacement()
  // and ReplaceMacros() handle an unclosed function-like macro reference
  // by terminating macro replacement at the name of the FLM and returning
  // its index in the result.  This allows the recursive call sites in
  // MacroReplacement to append any remaining tokens in their inputs to
  // that result and try again.  All other Fortran preprocessors share this
  // behavior.
  std::optional<TokenSequence> MacroReplacement(const TokenSequence &,
      Prescanner &,
      std::optional<std::size_t> *partialFunctionLikeMacro = nullptr);

  // Implements a preprocessor directive.
  void Directive(const TokenSequence &, Prescanner &);

  void PrintMacros(llvm::raw_ostream &out) const;

private:
  enum class IsElseActive { No, Yes };
  enum class CanDeadElseAppear { No, Yes };

  CharBlock SaveTokenAsName(const CharBlock &);
  TokenSequence ReplaceMacros(const TokenSequence &, Prescanner &,
      std::optional<std::size_t> *partialFunctionLikeMacro = nullptr);
  void SkipDisabledConditionalCode(
      const std::string &, IsElseActive, Prescanner &, ProvenanceRange);
  bool IsIfPredicateTrue(const TokenSequence &expr, std::size_t first,
      std::size_t exprTokens, Prescanner &);
  void LineDirective(const TokenSequence &, std::size_t, Prescanner &);

  AllSources &allSources_;
  std::list<std::string> names_;
  std::unordered_map<CharBlock, Definition> definitions_;
  std::stack<CanDeadElseAppear> ifStack_;
};
} // namespace Fortran::parser
#endif // FORTRAN_PARSER_PREPROCESSOR_H_
