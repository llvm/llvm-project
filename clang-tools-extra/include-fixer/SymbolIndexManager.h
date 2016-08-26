//===-- SymbolIndexManager.h - Managing multiple SymbolIndices --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEXMANAGER_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEXMANAGER_H

#include "SymbolIndex.h"
#include "find-all-symbols/SymbolInfo.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace include_fixer {

/// This class provides an interface for finding the header files corresponding
/// to an indentifier in the source code from multiple symbol databases.
class SymbolIndexManager {
public:
  void addSymbolIndex(std::unique_ptr<SymbolIndex> DB) {
    SymbolIndices.push_back(std::move(DB));
  }

  /// Search for header files to be included for an identifier.
  /// \param Identifier The identifier being searched for. May or may not be
  ///                   fully qualified.
  /// \param IsNestedSearch Whether searching nested classes. If true, the
  ///        method tries to strip identifier name parts from the end until it
  ///        finds the corresponding candidates in database (e.g for identifier
  ///        "b::foo", the method will try to find "b" if it fails to find
  ///        "b::foo").
  ///
  /// \returns A list of symbol candidates.
  std::vector<find_all_symbols::SymbolInfo>
  search(llvm::StringRef Identifier, bool IsNestedSearch = true) const;

private:
  std::vector<std::unique_ptr<SymbolIndex>> SymbolIndices;
};

} // namespace include_fixer
} // namespace clang

#endif
