//===-- SymbolIndex.h - Interface for symbol-header matching ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEX_H

#include "find-all-symbols/SymbolInfo.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {
namespace include_fixer {

/// This class provides an interface for finding all `SymbolInfo`s corresponding
/// to a symbol name from a symbol database.
class SymbolIndex {
public:
  virtual ~SymbolIndex() = default;

  /// Search for all `SymbolInfo`s corresponding to an identifier.
  /// \param Identifier The unqualified identifier being searched for.
  /// \returns A list of `SymbolInfo` candidates.
  // FIXME: Expose the type name so we can also insert using declarations (or
  // fix the usage)
  virtual std::vector<find_all_symbols::SymbolAndSignals>
  search(llvm::StringRef Identifier) = 0;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEX_H
