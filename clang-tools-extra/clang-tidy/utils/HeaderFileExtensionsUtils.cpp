//===--- HeaderFileExtensionsUtils.cpp - clang-tidy--------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HeaderFileExtensionsUtils.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace tidy {
namespace utils {

bool isExpansionLocInHeaderFile(
    SourceLocation Loc, const SourceManager &SM,
    const HeaderFileExtensionsSet &HeaderFileExtensions) {
  SourceLocation ExpansionLoc = SM.getExpansionLoc(Loc);
  StringRef FileExtension =
      llvm::sys::path::extension(SM.getFilename(ExpansionLoc));
  return HeaderFileExtensions.count(FileExtension.substr(1)) > 0;
}

bool isPresumedLocInHeaderFile(
    SourceLocation Loc, SourceManager &SM,
    const HeaderFileExtensionsSet &HeaderFileExtensions) {
  PresumedLoc PresumedLocation = SM.getPresumedLoc(Loc);
  StringRef FileExtension =
      llvm::sys::path::extension(PresumedLocation.getFilename());
  return HeaderFileExtensions.count(FileExtension.substr(1)) > 0;
}

bool isSpellingLocInHeaderFile(
    SourceLocation Loc, SourceManager &SM,
    const HeaderFileExtensionsSet &HeaderFileExtensions) {
  SourceLocation SpellingLoc = SM.getSpellingLoc(Loc);
  StringRef FileExtension =
      llvm::sys::path::extension(SM.getFilename(SpellingLoc));

  return HeaderFileExtensions.count(FileExtension.substr(1)) > 0;
}

bool parseHeaderFileExtensions(StringRef AllHeaderFileExtensions,
                               HeaderFileExtensionsSet &HeaderFileExtensions,
                               char delimiter) {
  SmallVector<StringRef, 5> Suffixes;
  AllHeaderFileExtensions.split(Suffixes, delimiter);
  HeaderFileExtensions.clear();
  for (StringRef Suffix : Suffixes) {
    StringRef Extension = Suffix.trim();
    for (StringRef::const_iterator it = Extension.begin();
         it != Extension.end(); ++it) {
      if (!isAlphanumeric(*it))
        return false;
    }
    HeaderFileExtensions.insert(Extension);
  }
  return true;
}

bool isHeaderFileExtension(StringRef FileName,
                           HeaderFileExtensionsSet HeaderFileExtensions) {
  StringRef extension = ::llvm::sys::path::extension(FileName);
  if (extension.startswith("."))
    extension = extension.substr(1);

  return HeaderFileExtensions.count(extension) > 0;
}

} // namespace utils
} // namespace tidy
} // namespace clang
