//===-- SymbolIndexManager.h - Managing multiple SymbolIndices --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEXMANAGER_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_SYMBOLINDEXMANAGER_H

#include "SymbolIndex.h"
#include "find-all-symbols/SymbolInfo.h"
#include "llvm/ADT/StringRef.h"

#ifdef _MSC_VER
// Disable warnings from ppltasks.h transitively included by <future>.
#pragma warning(push)
#pragma warning(disable:4530)
#endif

#include <future>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace clang {
namespace include_fixer {

/// This class provides an interface for finding the header files corresponding
/// to an identifier in the source code from multiple symbol databases.
class SymbolIndexManager {
public:
  void addSymbolIndex(std::function<std::unique_ptr<SymbolIndex>()> F) {
#if LLVM_ENABLE_THREADS
    auto Strategy = std::launch::async;
#else
    auto Strategy = std::launch::deferred;
#endif
    SymbolIndices.push_back(std::async(Strategy, F));
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
  search(llvm::StringRef Identifier, bool IsNestedSearch = true,
         llvm::StringRef FileName = "") const;

private:
  std::vector<std::shared_future<std::unique_ptr<SymbolIndex>>> SymbolIndices;
};

} // namespace include_fixer
} // namespace clang

#endif
