//===--- StdLib.h - Index the C and C++ standard library ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Eagerly indexing the standard library gives a much friendlier "warm start"
// with working code completion in a standalone file or small project.
//
// We act as if we saw a file which included the whole standard library:
//   #include <array>
//   #include <bitset>
//   #include <chrono>
//   ...
// We index this TU and feed the result into the dynamic index.
//
// This happens within the context of some particular open file, and we reuse
// its CompilerInvocation. Matching its include path, LangOpts etc ensures that
// we see the standard library and configuration that matches the project.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_STDLIB_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_STDLIB_H

#include "index/Symbol.h"
#include "support/ThreadsafeFS.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
class CompilerInvocation;
class LangOptions;
class HeaderSearch;
namespace clangd {

// The filesystem location where a standard library was found.
//
// This is the directory containing <vector> or <stdio.h>.
// It's used to ensure we only index files that are in the standard library.
//
// The paths are canonicalized (FS "real path" with symlinks resolved).
// This allows them to be easily compared against paths the indexer returns.
struct StdLibLocation {
  llvm::SmallVector<std::string> Paths;
};

// Tracks the state of standard library indexing within a particular index.
//
// In general, we don't want to index the standard library multiple times.
// In most cases, this class just acts as a flag to ensure we only do it once.
//
// However, if we first open a C++11 file, and then a C++20 file, we *do*
// want the index to be upgraded to include the extra symbols.
// Similarly, the C and C++ standard library can coexist.
class StdLibSet {
  std::atomic<int> Best[2] = {{-1}, {-1}};

public:
  // Determines if we should index the standard library in a configuration.
  //
  // This is true if:
  //  - standard library indexing is enabled for the file
  //  - the language version is higher than any previous add() for the language
  //  - the standard library headers exist on the search path
  // Returns the location where the standard library was found.
  //
  // This function is threadsafe.
  llvm::Optional<StdLibLocation> add(const LangOptions &, const HeaderSearch &);

  // Indicates whether a built index should be used.
  // It should not be used if a newer version has subsequently been added.
  //
  // Intended pattern is:
  //   if (add()) {
  //     symbols = indexStandardLibrary();
  //     if (isBest())
  //       index.update(symbols);
  //   }
  //
  // This is still technically racy: we could return true here, then another
  // thread could add->index->update a better library before we can update.
  // We'd then overwrite it with the older version.
  // However, it's very unlikely: indexing takes a long time.
  bool isBest(const LangOptions &) const;
};

// Index a standard library and return the discovered symbols.
//
// The compiler invocation should describe the file whose config we're reusing.
// We overwrite its virtual buffer with a lot of #include statements.
SymbolSlab indexStandardLibrary(std::unique_ptr<CompilerInvocation> Invocation,
                                const StdLibLocation &Loc,
                                const ThreadsafeFS &TFS);

// Variant that allows the umbrella header source to be specified.
// Exposed for testing.
SymbolSlab indexStandardLibrary(llvm::StringRef HeaderSources,
                                std::unique_ptr<CompilerInvocation> CI,
                                const StdLibLocation &Loc,
                                const ThreadsafeFS &TFS);

// Generate header containing #includes for all standard library headers.
// Exposed for testing.
llvm::StringRef getStdlibUmbrellaHeader(const LangOptions &);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_STDLIB_H
