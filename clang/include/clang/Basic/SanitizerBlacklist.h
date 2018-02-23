//===--- SanitizerBlacklist.h - Blacklist for sanitizers --------*- C++ -*-===//
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
#ifndef LLVM_CLANG_BASIC_SANITIZERBLACKLIST_H
#define LLVM_CLANG_BASIC_SANITIZERBLACKLIST_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SpecialCaseList.h"
#include <memory>

namespace clang {

class SanitizerBlacklist {
  std::unique_ptr<llvm::SpecialCaseList> SCL;
  SourceManager &SM;

public:
  SanitizerBlacklist(const std::vector<std::string> &BlacklistPaths,
                     SourceManager &SM);
  bool isBlacklistedGlobal(StringRef GlobalName,
                           StringRef Category = StringRef()) const;
  bool isBlacklistedType(StringRef MangledTypeName,
                         StringRef Category = StringRef()) const;
  bool isBlacklistedFunction(StringRef FunctionName) const;
  bool isBlacklistedFile(StringRef FileName,
                         StringRef Category = StringRef()) const;
  bool isBlacklistedLocation(SourceLocation Loc,
                             StringRef Category = StringRef()) const;
};

}  // end namespace clang

#endif
