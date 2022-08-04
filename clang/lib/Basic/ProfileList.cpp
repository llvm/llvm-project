//===--- ProfileList.h - ProfileList filter ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// User-provided filters include/exclude profile instrumentation in certain
// functions or files.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/ProfileList.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/SpecialCaseList.h"

#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace clang {

class ProfileSpecialCaseList : public llvm::SpecialCaseList {
public:
  static std::unique_ptr<ProfileSpecialCaseList>
  create(const std::vector<std::string> &Paths, llvm::vfs::FileSystem &VFS,
         std::string &Error);

  static std::unique_ptr<ProfileSpecialCaseList>
  createOrDie(const std::vector<std::string> &Paths,
              llvm::vfs::FileSystem &VFS);

  bool isEmpty() const { return Sections.empty(); }

  bool hasPrefix(StringRef Prefix) const {
    for (auto &SectionIter : Sections)
      if (SectionIter.Entries.count(Prefix) > 0)
        return true;
    return false;
  }
};

std::unique_ptr<ProfileSpecialCaseList>
ProfileSpecialCaseList::create(const std::vector<std::string> &Paths,
                               llvm::vfs::FileSystem &VFS,
                               std::string &Error) {
  auto PSCL = std::make_unique<ProfileSpecialCaseList>();
  if (PSCL->createInternal(Paths, VFS, Error))
    return PSCL;
  return nullptr;
}

std::unique_ptr<ProfileSpecialCaseList>
ProfileSpecialCaseList::createOrDie(const std::vector<std::string> &Paths,
                                    llvm::vfs::FileSystem &VFS) {
  std::string Error;
  if (auto PSCL = create(Paths, VFS, Error))
    return PSCL;
  llvm::report_fatal_error(llvm::Twine(Error));
}

}

ProfileList::ProfileList(ArrayRef<std::string> Paths, SourceManager &SM)
    : SCL(ProfileSpecialCaseList::createOrDie(
          Paths, SM.getFileManager().getVirtualFileSystem())),
      Empty(SCL->isEmpty()),
      Default(SCL->hasPrefix("fun") || SCL->hasPrefix("src")), SM(SM) {}

ProfileList::~ProfileList() = default;

static StringRef getSectionName(CodeGenOptions::ProfileInstrKind Kind) {
  switch (Kind) {
  case CodeGenOptions::ProfileNone:
    return "";
  case CodeGenOptions::ProfileClangInstr:
    return "clang";
  case CodeGenOptions::ProfileIRInstr:
    return "llvm";
  case CodeGenOptions::ProfileCSIRInstr:
    return "csllvm";
  }
  llvm_unreachable("Unhandled CodeGenOptions::ProfileInstrKind enum");
}

llvm::Optional<bool>
ProfileList::isFunctionExcluded(StringRef FunctionName,
                                CodeGenOptions::ProfileInstrKind Kind) const {
  StringRef Section = getSectionName(Kind);
  if (SCL->inSection(Section, "!fun", FunctionName))
    return true;
  if (SCL->inSection(Section, "fun", FunctionName))
    return false;
  return None;
}

llvm::Optional<bool>
ProfileList::isLocationExcluded(SourceLocation Loc,
                                CodeGenOptions::ProfileInstrKind Kind) const {
  return isFileExcluded(SM.getFilename(SM.getFileLoc(Loc)), Kind);
}

llvm::Optional<bool>
ProfileList::isFileExcluded(StringRef FileName,
                            CodeGenOptions::ProfileInstrKind Kind) const {
  StringRef Section = getSectionName(Kind);
  if (SCL->inSection(Section, "!src", FileName))
    return true;
  if (SCL->inSection(Section, "src", FileName))
    return false;
  return None;
}
