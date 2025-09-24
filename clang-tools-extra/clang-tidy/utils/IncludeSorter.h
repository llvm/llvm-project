//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDESORTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDESORTER_H

#include "../ClangTidyCheck.h"
#include <optional>
#include <string>

namespace clang::tidy {
namespace utils {

/// Class used by ``IncludeInserterCallback`` to record the names of the
/// inclusions in a given source file being processed and generate the necessary
/// commands to sort the inclusions according to the precedence encoded in
/// ``IncludeKinds``.
class IncludeSorter {
public:
  /// Supported include styles.
  enum IncludeStyle { IS_LLVM = 0, IS_Google = 1, IS_Google_ObjC };

  /// The classifications of inclusions, in the order they should be sorted.
  enum IncludeKinds {
    IK_MainTUInclude = 0,    ///< e.g. ``#include "foo.h"`` when editing foo.cc
    IK_CSystemInclude = 1,   ///< e.g. ``#include <stdio.h>``
    IK_CXXSystemInclude = 2, ///< e.g. ``#include <vector>``
    IK_NonSystemInclude = 3, ///< e.g. ``#include "bar.h"``
    IK_GeneratedInclude = 4, ///< e.g. ``#include "bar.proto.h"``
    IK_InvalidInclude = 5    ///< total number of valid ``IncludeKind``s
  };

  /// ``IncludeSorter`` constructor; takes the FileID and name of the file to be
  /// processed by the sorter.
  IncludeSorter(const SourceManager *SourceMgr, const FileID FileID,
                StringRef FileName, IncludeStyle Style);

  /// Adds the given include directive to the sorter.
  void addInclude(StringRef FileName, bool IsAngled,
                  SourceLocation HashLocation, SourceLocation EndLocation);

  /// Creates a quoted inclusion directive in the right sort order. Returns
  /// std::nullopt on error or if header inclusion directive for header already
  /// exists.
  std::optional<FixItHint> createIncludeInsertion(StringRef FileName,
                                                  bool IsAngled);

private:
  using SourceRangeVector = SmallVector<SourceRange, 1>;

  const SourceManager *SourceMgr;
  const IncludeStyle Style;
  FileID CurrentFileID;
  /// The file name stripped of common suffixes.
  StringRef CanonicalFile;
  /// Locations of visited include directives.
  SourceRangeVector SourceLocations;
  /// Mapping from file name to #include locations.
  llvm::StringMap<SourceRangeVector> IncludeLocations;
  /// Includes sorted into buckets.
  SmallVector<std::string, 1> IncludeBucket[IK_InvalidInclude];
};

} // namespace utils

template <> struct OptionEnumMapping<utils::IncludeSorter::IncludeStyle> {
  static ArrayRef<std::pair<utils::IncludeSorter::IncludeStyle, StringRef>>
  getEnumMapping();
};
} // namespace clang::tidy
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDESORTER_H
