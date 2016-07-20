//===------------ IncludeSorter.h - clang-tidy ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDESORTER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDESORTER_H

#include "../ClangTidy.h"
#include <string>

namespace clang {
namespace tidy {
namespace utils {

/// Class used by ``IncludeInserterCallback`` to record the names of the
/// inclusions in a given source file being processed and generate the necessary
/// commands to sort the inclusions according to the precedence encoded in
/// ``IncludeKinds``.
class IncludeSorter {
public:
  /// Supported include styles.
  enum IncludeStyle { IS_LLVM = 0, IS_Google = 1 };

  /// Converts "llvm" to ``IS_LLVM``, otherwise returns ``IS_Google``.
  static IncludeStyle parseIncludeStyle(const std::string &Value);

  /// Converts ``IncludeStyle`` to string representation.
  static StringRef toString(IncludeStyle Style);

  /// The classifications of inclusions, in the order they should be sorted.
  enum IncludeKinds {
    IK_MainTUInclude = 0,    ///< e.g. ``#include "foo.h"`` when editing foo.cc
    IK_CSystemInclude = 1,   ///< e.g. ``#include <stdio.h>``
    IK_CXXSystemInclude = 2, ///< e.g. ``#include <vector>``
    IK_NonSystemInclude = 3, ///< e.g. ``#include "bar.h"``
    IK_InvalidInclude = 4    ///< total number of valid ``IncludeKind``s
  };

  /// ``IncludeSorter`` constructor; takes the FileID and name of the file to be
  /// processed by the sorter.
  IncludeSorter(const SourceManager *SourceMgr, const LangOptions *LangOpts,
                const FileID FileID, StringRef FileName, IncludeStyle Style);

  /// Returns the ``SourceManager``-specific file ID for the file being handled
  /// by the sorter.
  const FileID current_FileID() const { return CurrentFileID; }

  /// Adds the given include directive to the sorter.
  void AddInclude(StringRef FileName, bool IsAngled,
                  SourceLocation HashLocation, SourceLocation EndLocation);

  /// Returns the edits needed to sort the current set of includes and reset the
  /// internal state (so that different blocks of includes are sorted separately
  /// within the same file).
  std::vector<FixItHint> GetEdits();

  /// Creates a quoted inclusion directive in the right sort order. Returns None
  /// on error or if header inclusion directive for header already exists.
  Optional<FixItHint> CreateIncludeInsertion(StringRef FileName, bool IsAngled);

private:
  typedef SmallVector<SourceRange, 1> SourceRangeVector;

  const SourceManager *SourceMgr;
  const LangOptions *LangOpts;
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
} // namespace tidy
} // namespace clang
#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_INCLUDESORTER_H
