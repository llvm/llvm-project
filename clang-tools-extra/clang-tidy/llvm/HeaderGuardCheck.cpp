//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderGuardCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"

namespace clang::tidy::llvm_check {

LLVMHeaderGuardCheck::LLVMHeaderGuardCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : HeaderGuardCheck(Name, Context),
      HeaderDirs(utils::options::parseStringList(
          Options.get("HeaderDirs", "include"))) {}

std::string LLVMHeaderGuardCheck::getHeaderGuard(StringRef Filename,
                                                 StringRef OldGuard) {
  std::string Guard = tooling::getAbsolutePath(Filename);

  // When running under Windows, need to convert the path separators from
  // `\` to `/`.
  Guard = llvm::sys::path::convert_to_slash(Guard);

  // Sanitize the path. There are some rules for compatibility with the historic
  // style in include/llvm and include/clang which we want to preserve.

  // consider all directories from HeaderDirs option. Stop at first found.
  for (StringRef HeaderDir : HeaderDirs) {
    size_t PosHeaderDir = Guard.rfind(HeaderDir.str() + "/");
    if (PosHeaderDir != StringRef::npos) {
      // We don't want the header dir in our guards, i.e. _INCLUDE_
      Guard = Guard.substr(PosHeaderDir + HeaderDir.size() + 1);
      break; // stop at first found
    }
  }

  // For clang we drop the _TOOLS_.
  const size_t PosToolsClang = Guard.rfind("tools/clang/");
  if (PosToolsClang != StringRef::npos)
    Guard = Guard.substr(PosToolsClang + std::strlen("tools/"));

  // Unlike LLVM svn, LLVM git monorepo is named llvm-project, so we replace
  // "/llvm-project/" with the canonical "/llvm/".
  const static StringRef LLVMProject = "/llvm-project/";
  const size_t PosLLVMProject = Guard.rfind(LLVMProject);
  if (PosLLVMProject != StringRef::npos)
    Guard = Guard.replace(PosLLVMProject, LLVMProject.size(), "/llvm/");

  // The remainder is LLVM_FULL_PATH_TO_HEADER_H
  const size_t PosLLVM = Guard.rfind("llvm/");
  if (PosLLVM != StringRef::npos)
    Guard = Guard.substr(PosLLVM);

  llvm::replace(Guard, '/', '_');
  llvm::replace(Guard, '.', '_');
  llvm::replace(Guard, '-', '_');

  // The prevalent style in clang is LLVM_CLANG_FOO_BAR_H
  if (StringRef(Guard).starts_with("clang"))
    Guard = "LLVM_" + Guard;

  // The prevalent style in flang is FORTRAN_FOO_BAR_H
  if (StringRef(Guard).starts_with("flang"))
    Guard = "FORTRAN" + Guard.substr(sizeof("flang") - 1);

  return StringRef(Guard).upper();
}

void LLVMHeaderGuardCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HeaderDirs",
                utils::options::serializeStringList(HeaderDirs));
}

} // namespace clang::tidy::llvm_check
