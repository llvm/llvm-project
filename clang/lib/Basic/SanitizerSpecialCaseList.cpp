//===--- SanitizerSpecialCaseList.cpp - SCL for sanitizers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An extension of SpecialCaseList to allowing querying sections by
// SanitizerMask.
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/SanitizerSpecialCaseList.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang;

std::unique_ptr<SanitizerSpecialCaseList>
SanitizerSpecialCaseList::create(const std::vector<std::string> &Paths,
                                 llvm::vfs::FileSystem &VFS,
                                 std::string &Error) {
  std::unique_ptr<clang::SanitizerSpecialCaseList> SSCL(
      new SanitizerSpecialCaseList());
  if (SSCL->createInternal(Paths, VFS, Error)) {
    SSCL->createSanitizerSections();
    return SSCL;
  }
  return nullptr;
}

std::unique_ptr<SanitizerSpecialCaseList>
SanitizerSpecialCaseList::createOrDie(const std::vector<std::string> &Paths,
                                      llvm::vfs::FileSystem &VFS) {
  std::string Error;
  if (auto SSCL = create(Paths, VFS, Error))
    return SSCL;
  llvm::report_fatal_error(StringRef(Error));
}

void SanitizerSpecialCaseList::createSanitizerSections() {
  for (const auto &S : Sections) {
    SanitizerMask Mask;

#define SANITIZER(NAME, ID)                                                    \
  if (S.SectionMatcher.matchAny(NAME))                                         \
    Mask |= SanitizerKind::ID;
#define SANITIZER_GROUP(NAME, ID, ALIAS) SANITIZER(NAME, ID)

#include "clang/Basic/Sanitizers.def"
#undef SANITIZER
#undef SANITIZER_GROUP

    SanitizerSections.emplace_back(Mask, S);
  }
}

bool SanitizerSpecialCaseList::inSection(SanitizerMask Mask, StringRef Prefix,
                                         StringRef Query,
                                         StringRef Category) const {
  return inSectionBlame(Mask, Prefix, Query, Category) != NotFound;
}

std::pair<unsigned, unsigned>
SanitizerSpecialCaseList::inSectionBlame(SanitizerMask Mask, StringRef Prefix,
                                         StringRef Query,
                                         StringRef Category) const {
  for (const auto &S : llvm::reverse(SanitizerSections)) {
    if (S.Mask & Mask) {
      unsigned LineNum = S.S.getLastMatch(Prefix, Query, Category);
      if (LineNum > 0)
        return {S.S.FileIdx, LineNum};
    }
  }
  return NotFound;
}
