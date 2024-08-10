//===------------------ ProjectModules.h -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROJECTMODULES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PROJECTMODULES_H

#include "support/Path.h"
#include "support/ThreadsafeFS.h"

#include <memory>

namespace clang {
namespace clangd {

/// An interface to query the modules information in the project.
/// Users should get instances of `ProjectModules` from
/// `GlobalCompilationDatabase::getProjectModules(PathRef)`.
///
/// Currently, the modules information includes:
/// - Given a source file, what are the required modules.
/// - Given a module name and a required source file, what is
///   the corresponding source file.
///
/// Note that there can be multiple source files declaring the same module
/// in a valid project. Although the language specification requires that
/// every module unit's name must be unique in valid program, there can be
/// multiple program in a project. And it is technically valid if these program
/// doesn't interfere with each other.
///
/// A module name should be in the format:
/// `<primary-module-name>[:partition-name]`. So module names covers partitions.
class ProjectModules {
public:
  virtual std::vector<std::string> getRequiredModules(PathRef File) = 0;
  virtual PathRef
  getSourceForModuleName(llvm::StringRef ModuleName,
                         PathRef RequiredSrcFile = PathRef()) = 0;

  virtual ~ProjectModules() = default;
};

} // namespace clangd
} // namespace clang

#endif
