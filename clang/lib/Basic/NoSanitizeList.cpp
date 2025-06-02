//===--- NoSanitizeList.cpp - Ignored list for sanitizers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided ignore-list used to disable/alter instrumentation done in
// sanitizers.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/NoSanitizeList.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SanitizerSpecialCaseList.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

NoSanitizeList::NoSanitizeList(const std::vector<std::string> &NoSanitizePaths,
                               SourceManager &SM)
    : SSCL(SanitizerSpecialCaseList::createOrDie(
          NoSanitizePaths, SM.getFileManager().getVirtualFileSystem())),
      SM(SM) {}

NoSanitizeList::~NoSanitizeList() = default;

bool NoSanitizeList::containsPrefix(SanitizerMask Mask, StringRef Prefix,
                                    StringRef Name, StringRef Category) const {
  std::pair<unsigned, unsigned> NoSan =
      SSCL->inSectionBlame(Mask, Prefix, Name, Category);
  if (NoSan == llvm::SpecialCaseList::NotFound)
    return false;
  std::pair<unsigned, unsigned> San =
      SSCL->inSectionBlame(Mask, Prefix, Name, "sanitize");
  // The statement evaluates to true under the following conditions:
  // 1. The string "prefix:*=sanitize" is absent.
  // 2. If "prefix:*=sanitize" is present, its (File Index, Line Number) is less
  // than that of "prefix:*".
  return San == llvm::SpecialCaseList::NotFound || NoSan > San;
}

bool NoSanitizeList::containsGlobal(SanitizerMask Mask, StringRef GlobalName,
                                    StringRef Category) const {
  return containsPrefix(Mask, "global", GlobalName, Category);
}

bool NoSanitizeList::containsType(SanitizerMask Mask, StringRef MangledTypeName,
                                  StringRef Category) const {
  return containsPrefix(Mask, "type", MangledTypeName, Category);
}

bool NoSanitizeList::containsFunction(SanitizerMask Mask,
                                      StringRef FunctionName) const {
  return containsPrefix(Mask, "fun", FunctionName, {});
}

bool NoSanitizeList::containsFile(SanitizerMask Mask, StringRef FileName,
                                  StringRef Category) const {
  return containsPrefix(Mask, "src", FileName, Category);
}

bool NoSanitizeList::containsMainFile(SanitizerMask Mask, StringRef FileName,
                                      StringRef Category) const {
  return containsPrefix(Mask, "mainfile", FileName, Category);
}

bool NoSanitizeList::containsLocation(SanitizerMask Mask, SourceLocation Loc,
                                      StringRef Category) const {
  return Loc.isValid() &&
         containsFile(Mask, SM.getFilename(SM.getFileLoc(Loc)), Category);
}
