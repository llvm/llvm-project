//===--- FileExtensionsUtils.h - clang-tidy --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FILE_EXTENSIONS_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FILE_EXTENSIONS_UTILS_H

#include "../FileExtensionsSet.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace clang::tidy::utils {

/// Checks whether expansion location of \p Loc is in header file.
bool isExpansionLocInHeaderFile(SourceLocation Loc, const SourceManager &SM,
                                const FileExtensionsSet &HeaderFileExtensions);

/// Checks whether presumed location of \p Loc is in header file.
bool isPresumedLocInHeaderFile(SourceLocation Loc, SourceManager &SM,
                               const FileExtensionsSet &HeaderFileExtensions);

/// Checks whether spelling location of \p Loc is in header file.
bool isSpellingLocInHeaderFile(SourceLocation Loc, SourceManager &SM,
                               const FileExtensionsSet &HeaderFileExtensions);

/// Returns recommended default value for the list of header file
/// extensions.
inline StringRef defaultHeaderFileExtensions() { return ";h;hh;hpp;hxx"; }

/// Returns recommended default value for the list of implementation file
/// extensions.
inline StringRef defaultImplementationFileExtensions() {
  return "c;cc;cpp;cxx";
}

/// Returns recommended default value for the list of file extension
/// delimiters.
inline StringRef defaultFileExtensionDelimiters() { return ",;"; }

/// Parses header file extensions from a semicolon-separated list.
bool parseFileExtensions(StringRef AllFileExtensions,
                         FileExtensionsSet &FileExtensions,
                         StringRef Delimiters);

/// Decides whether a file has a header file extension.
/// Returns the file extension, if included in the provided set.
std::optional<StringRef>
getFileExtension(StringRef FileName, const FileExtensionsSet &FileExtensions);

/// Decides whether a file has one of the specified file extensions.
bool isFileExtension(StringRef FileName,
                     const FileExtensionsSet &FileExtensions);

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_FILE_EXTENSIONS_UTILS_H
