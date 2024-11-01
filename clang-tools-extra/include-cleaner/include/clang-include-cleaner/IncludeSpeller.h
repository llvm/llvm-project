//===--- IncludeSpeller.h - Spelling strategies for headers.-------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// An extension point to let applications introduce custom spelling
// strategies for physical headers.
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_INCLUDESPELLER_H
#define CLANG_INCLUDE_CLEANER_INCLUDESPELLER_H

#include "clang-include-cleaner/Types.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/Support/Registry.h"
#include <string>

namespace clang::include_cleaner {

/// IncludeSpeller provides an extension point to allow clients implement
/// custom include spelling strategies for physical headers.
class IncludeSpeller {
public:
  /// Provides the necessary information for custom spelling computations.
  struct Input {
    const Header &H;
    const HeaderSearch &HS;
    const FileEntry *Main;
  };
  virtual ~IncludeSpeller() = default;

  /// Takes in an `Input` struct with necessary infos about a header and
  /// returns a verbatim include spelling (with angles/quotes) or an empty
  /// string to indicate no customizations are needed.
  virtual std::string operator()(const Input &Input) const = 0;
};

using IncludeSpellingStrategy = llvm::Registry<IncludeSpeller>;

/// Generates a spelling for the header in the `Input` that can be directly
/// included in the main file. When the `Input` specifies a physical header,
/// prefers the spelling provided by custom llvm strategies, if any.
/// Otherwise, uses header search info to generate shortest spelling.
std::string spellHeader(const IncludeSpeller::Input &Input);
} // namespace clang::include_cleaner

#endif
