//===--- FileExtensionsUtils.cpp - clang-tidy -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FileExtensionsUtils.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace tidy {
namespace utils {

bool isExpansionLocInHeaderFile(SourceLocation Loc, const SourceManager &SM,
                                const FileExtensionsSet &HeaderFileExtensions) {
  SourceLocation ExpansionLoc = SM.getExpansionLoc(Loc);
  return isFileExtension(SM.getFilename(ExpansionLoc), HeaderFileExtensions);
}

bool isPresumedLocInHeaderFile(SourceLocation Loc, SourceManager &SM,
                               const FileExtensionsSet &HeaderFileExtensions) {
  PresumedLoc PresumedLocation = SM.getPresumedLoc(Loc);
  return isFileExtension(PresumedLocation.getFilename(), HeaderFileExtensions);
}

bool isSpellingLocInHeaderFile(SourceLocation Loc, SourceManager &SM,
                               const FileExtensionsSet &HeaderFileExtensions) {
  SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);
  return isFileExtension(SM.getFilename(SpellingLoc), HeaderFileExtensions);
}

bool parseFileExtensions(StringRef AllFileExtensions,
                         FileExtensionsSet &FileExtensions,
                         StringRef Delimiters) {
  SmallVector<StringRef, 5> Suffixes;
  for (char Delimiter : Delimiters) {
    if (AllFileExtensions.contains(Delimiter)) {
      AllFileExtensions.split(Suffixes, Delimiter);
      break;
    }
  }

  FileExtensions.clear();
  for (StringRef Suffix : Suffixes) {
    StringRef Extension = Suffix.trim();
    if (!llvm::all_of(Extension, isAlphanumeric))
      return false;
    FileExtensions.insert(Extension);
  }
  return true;
}

llvm::Optional<StringRef>
getFileExtension(StringRef FileName, const FileExtensionsSet &FileExtensions) {
  StringRef Extension = llvm::sys::path::extension(FileName);
  if (Extension.empty())
    return llvm::None;
  // Skip "." prefix.
  if (!FileExtensions.count(Extension.substr(1)))
    return llvm::None;
  return Extension;
}

bool isFileExtension(StringRef FileName,
                     const FileExtensionsSet &FileExtensions) {
  return getFileExtension(FileName, FileExtensions).has_value();
}

} // namespace utils
} // namespace tidy
} // namespace clang
