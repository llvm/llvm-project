//===--- HeaderFileExtensionsUtils.h - clang-tidy----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADER_FILE_EXTENSIONS_UTILS_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADER_FILE_EXTENSIONS_UTILS_H

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace tidy {
namespace utils {

typedef llvm::SmallSet<llvm::StringRef, 5> HeaderFileExtensionsSet;

/// \brief Checks whether expansion location of \p Loc is in header file.
bool isExpansionLocInHeaderFile(
    SourceLocation Loc, const SourceManager &SM,
    const HeaderFileExtensionsSet &HeaderFileExtensions);

/// \brief Checks whether presumed location of \p Loc is in header file.
bool isPresumedLocInHeaderFile(
    SourceLocation Loc, SourceManager &SM,
    const HeaderFileExtensionsSet &HeaderFileExtensions);

/// \brief Checks whether spelling location of \p Loc is in header file.
bool isSpellingLocInHeaderFile(
    SourceLocation Loc, SourceManager &SM,
    const HeaderFileExtensionsSet &HeaderFileExtensions);

/// \brief Returns recommended default value for the list of header file
/// extensions.
inline StringRef defaultHeaderFileExtensions() { return ",h,hh,hpp,hxx"; }

/// \brief Parses header file extensions from a semicolon-separated list.
bool parseHeaderFileExtensions(StringRef AllHeaderFileExtensions,
                               HeaderFileExtensionsSet &HeaderFileExtensions,
                               char delimiter);

/// \brief Decides whether a file has a header file extension.
bool isHeaderFileExtension(StringRef FileName,
                           const HeaderFileExtensionsSet &HeaderFileExtensions);

} // namespace utils
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_HEADER_FILE_EXTENSIONS_UTILS_H
