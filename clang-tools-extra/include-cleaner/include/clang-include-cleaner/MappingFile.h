//===--- MappingFile.h - IWYU mapping file support ------------------C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for IWYU mapping files (.imp).
// Format:
// https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/IWYUMappings.md
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INCLUDE_CLEANER_MAPPINGFILE_H
#define CLANG_INCLUDE_CLEANER_MAPPINGFILE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <string>
#include <utility>
#include <vector>

namespace clang::include_cleaner {

/// Parsed data from IWYU mapping files (.imp).
///
/// Mapping files declare two kinds of relationships:
///  - Include mappings: "when a symbol comes from <private.h>, use <public.h>
///    instead."
///  - Symbol mappings: "when symbol Foo is referenced, include <foo.h>."
struct MappingFile {
  /// Maps a private header path (bare, no brackets) to the public header
  /// spelling that should be used instead, e.g. "foo/detail.h" -> "<foo.h>".
  llvm::StringMap<std::string> IncludeMappings;

  /// Maps a symbol name (possibly qualified) to the header spelling that
  /// should be included for it, e.g. "NULL" -> "<stddef.h>".
  llvm::StringMap<std::string> SymbolMappings;

  /// Regex patterns for include mappings (from "@<...>" entries).
  /// Each entry is (raw_regex, public_header_spelling).
  /// Patterns are matched against path suffixes, e.g. "AE/.*" from "@<AE/.*>".
  std::vector<std::pair<std::string, std::string>> IncludeRegexPatterns;

  /// Merges \p Other into this mapping. For duplicate keys, \p Other wins.
  void merge(MappingFile Other);
};

/// Parse one or more IWYU mapping files (.imp) into \p Result.
///
/// Each file is a YAML array of objects with "include", "symbol", or "ref"
/// keys.  The format is JSON-compatible YAML: supports unquoted private/public
/// visibility values, trailing commas, and # line comments.
/// "ref" entries are resolved relative to the file that contains them.
///
/// Multiple files are merged; later entries for the same key win.
/// Returns an error if any file cannot be read or has invalid syntax.
llvm::Expected<MappingFile>
parseMappingFiles(llvm::ArrayRef<std::string> Paths);

} // namespace clang::include_cleaner

#endif // CLANG_INCLUDE_CLEANER_MAPPINGFILE_H
