//===--- SanitizerBlacklist.cpp - Blacklist for sanitizers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided blacklist used to disable/alter instrumentation done in
// sanitizers.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SanitizerBlacklist.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SanitizerSpecialCaseList.h"
#include "clang/Basic/Sanitizers.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;

SanitizerBlacklist::SanitizerBlacklist(
    const std::vector<std::string> &BlacklistPaths, SourceManager &SM)
    : SSCL(SanitizerSpecialCaseList::createOrDie(
          BlacklistPaths, SM.getFileManager().getVirtualFileSystem())),
      SM(SM) {}

SanitizerBlacklist::~SanitizerBlacklist() = default;

bool SanitizerBlacklist::isBlacklistedGlobal(SanitizerMask Mask,
                                             StringRef GlobalName,
                                             StringRef Category) const {
  return SSCL->inSection(Mask, "global", GlobalName, Category);
}

bool SanitizerBlacklist::isBlacklistedType(SanitizerMask Mask,
                                           StringRef MangledTypeName,
                                           StringRef Category) const {
  return SSCL->inSection(Mask, "type", MangledTypeName, Category);
}

bool SanitizerBlacklist::isBlacklistedFunction(SanitizerMask Mask,
                                               StringRef FunctionName) const {
  return SSCL->inSection(Mask, "fun", FunctionName);
}

bool SanitizerBlacklist::isBlacklistedFile(SanitizerMask Mask,
                                           StringRef FileName,
                                           StringRef Category) const {
  return SSCL->inSection(Mask, "src", FileName, Category);
}

bool SanitizerBlacklist::isBlacklistedLocation(SanitizerMask Mask,
                                               SourceLocation Loc,
                                               StringRef Category) const {
  return Loc.isValid() &&
         isBlacklistedFile(Mask, SM.getFilename(SM.getFileLoc(Loc)), Category);
}

