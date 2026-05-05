//===--- HeaderFilterHelpers.h - clang-tidy header filtering ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HEADERFILTERHELPERS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HEADERFILTERHELPERS_H

#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"

namespace clang::tidy {

/// Evaluates clang-tidy's header filters for source locations and caches the
/// per-file results for a translation unit.
class HeaderFilterLocationFilter {
public:
  HeaderFilterLocationFilter(llvm::StringRef HeaderFilterRegex,
                             llvm::StringRef ExcludeHeaderFilterRegex)
      : HeaderFilter(HeaderFilterRegex),
        ExcludeHeaderFilter(ExcludeHeaderFilterRegex) {}

  /// Returns true when the location should be treated as in scope for
  /// clang-tidy's header filters.
  ///
  /// Main-file locations are always in scope. Invalid locations and locations
  /// without a FileEntry (such as command-line buffers) are also treated as in
  /// scope to match clang-tidy's diagnostic filtering behavior.
  bool shouldInclude(SourceLocation Location, const SourceManager &Sources) {
    if (!Location.isValid())
      return true;

    if (Sources.isInMainFile(Location))
      return true;

    const FileID FID = Sources.getDecomposedExpansionLoc(Location).first;
    if (const auto It = Cache.find(FID); It != Cache.end())
      return It->second;

    bool Result = true;
    if (const OptionalFileEntryRef File = Sources.getFileEntryRefForID(FID)) {
      const llvm::StringRef FileName = File->getName();
      Result =
          HeaderFilter.match(FileName) && !ExcludeHeaderFilter.match(FileName);
    }

    Cache[FID] = Result;
    return Result;
  }

private:
  llvm::Regex HeaderFilter;
  llvm::Regex ExcludeHeaderFilter;
  llvm::DenseMap<FileID, bool> Cache;
};

} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_HEADERFILTERHELPERS_H
