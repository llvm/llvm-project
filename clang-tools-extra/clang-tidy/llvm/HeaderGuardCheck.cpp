//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderGuardCheck.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace clang::tidy::llvm_check {

LLVMHeaderGuardCheck::LLVMHeaderGuardCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : HeaderGuardCheck(Name, Context) {}

// Attempt to find the root of the LLVM project monorepo by walking up the
// directory tree from Filename and looking for a ".git" file/directory.
// This allows us to find the root even when working in git worktrees with
// arbitrary names.
std::string
LLVMHeaderGuardCheck::findLLVMProjectRoot(StringRef Filename) const {
  SmallString<256> Path = Filename;
  SmallString<256> Parent = llvm::sys::path::parent_path(Path);
  while (!Parent.empty() && Parent != Path) {
    Path = Parent;

    // Check for .git (file or directory) which indicates the root of the
    // git repository or worktree.
    SmallString<256> GitPath = Path;
    llvm::sys::path::append(GitPath, ".git");
    if (llvm::sys::fs::exists(GitPath))
      return std::string(Path);

    Parent = llvm::sys::path::parent_path(Path);
  }
  return "";
}

std::string LLVMHeaderGuardCheck::getHeaderGuard(StringRef Filename,
                                                 StringRef OldGuard) {
  const std::string AbsolutePath = tooling::getAbsolutePath(Filename);
  std::string Guard = AbsolutePath;

  // Check for "/llvm-project/" using a path with normalized slashes to ensure
  // Windows paths (which use backslashes) are matched correctly.
  const std::string CanonicalPath =
      llvm::sys::path::convert_to_slash(AbsolutePath);
  if (!StringRef(CanonicalPath).contains("/llvm-project/")) {
    const std::string Root = findLLVMProjectRoot(AbsolutePath);
    if (!Root.empty()) {
      StringRef RelativePath = StringRef(AbsolutePath).substr(Root.size());
      if (!RelativePath.empty() &&
          llvm::sys::path::is_separator(RelativePath.front()))
        RelativePath = RelativePath.drop_front();
      Guard = ("/llvm-project/" + RelativePath).str();
    }
  }

  // When running under Windows, need to convert the path separators from
  // `\` to `/`.
  Guard = llvm::sys::path::convert_to_slash(Guard);

  // Sanitize the path. There are some rules for compatibility with the historic
  // style in include/llvm and include/clang which we want to preserve.

  // We don't want _INCLUDE_ in our guards.
  const size_t PosInclude = Guard.rfind("include/");
  if (PosInclude != StringRef::npos)
    Guard = Guard.substr(PosInclude + std::strlen("include/"));

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

} // namespace clang::tidy::llvm_check
