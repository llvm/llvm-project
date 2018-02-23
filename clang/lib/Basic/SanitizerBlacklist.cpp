//===--- SanitizerBlacklist.cpp - Blacklist for sanitizers ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// User-provided blacklist used to disable/alter instrumentation done in
// sanitizers.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/SanitizerBlacklist.h"

using namespace clang;

SanitizerBlacklist::SanitizerBlacklist(
    const std::vector<std::string> &BlacklistPaths, SourceManager &SM)
    : SCL(llvm::SpecialCaseList::createOrDie(BlacklistPaths)), SM(SM) {}

bool SanitizerBlacklist::isBlacklistedGlobal(StringRef GlobalName,
                                             StringRef Category) const {
  return SCL->inSection("global", GlobalName, Category);
}

bool SanitizerBlacklist::isBlacklistedType(StringRef MangledTypeName,
                                           StringRef Category) const {
  return SCL->inSection("type", MangledTypeName, Category);
}

bool SanitizerBlacklist::isBlacklistedFunction(StringRef FunctionName) const {
  return SCL->inSection("fun", FunctionName);
}

bool SanitizerBlacklist::isBlacklistedFile(StringRef FileName,
                                           StringRef Category) const {
  return SCL->inSection("src", FileName, Category);
}

bool SanitizerBlacklist::isBlacklistedLocation(SourceLocation Loc,
                                               StringRef Category) const {
  return Loc.isValid() &&
         isBlacklistedFile(SM.getFilename(SM.getFileLoc(Loc)), Category);
}

