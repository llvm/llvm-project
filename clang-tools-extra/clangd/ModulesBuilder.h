//===----------------- ModulesBuilder.h --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Experimental support for C++20 Modules.
//
// Currently we simplify the implementations by preventing reusing module files
// across different versions and different source files. But this is clearly a
// waste of time and space in the end of the day.
//
// TODO: Supporting reusing module files across different versions and
// different source files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULES_BUILDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULES_BUILDER_H

#include "GlobalCompilationDatabase.h"
#include "ProjectModules.h"
#include "support/Path.h"
#include "support/ThreadsafeFS.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/SmallString.h"
#include <memory>

namespace clang {
namespace clangd {

/// Store all the needed module files information to parse a single
/// source file. e.g.,
///
///   ```
///   // a.cppm
///   export module a;
///
///   // b.cppm
///   export module b;
///   import a;
///
///   // c.cppm
///   export module c;
///   import b;
///   ```
///
/// For the source file `c.cppm`, an instance of the class will store
/// the module files for `a.cppm` and `b.cppm`. But the module file for `c.cppm`
/// won't be stored. Since it is not needed to parse `c.cppm`.
///
/// Users should only get PrerequisiteModules from
/// `ModulesBuilder::buildPrerequisiteModulesFor(...)`.
///
/// Users can detect whether the PrerequisiteModules is still up to date by
/// calling the `canReuse()` member function.
///
/// The users should call `adjustHeaderSearchOptions(...)` to update the
/// compilation commands to select the built module files first. Before calling
/// `adjustHeaderSearchOptions()`, users should call `canReuse()` first to check
/// if all the stored module files are valid. In case they are not valid,
/// users should call `ModulesBuilder::buildPrerequisiteModulesFor(...)` again
/// to get the new PrerequisiteModules.
class PrerequisiteModules {
public:
  /// Change commands to load the module files recorded in this
  /// PrerequisiteModules first.
  virtual void
  adjustHeaderSearchOptions(HeaderSearchOptions &Options) const = 0;

  /// Whether or not the built module files are up to date.
  /// Note that this can only be used after building the module files.
  virtual bool
  canReuse(const CompilerInvocation &CI,
           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) const = 0;

  virtual ~PrerequisiteModules() = default;
};

/// This class handles building module files for a given source file.
///
/// In the future, we want the class to manage the module files acorss
/// different versions and different source files.
class ModulesBuilder {
public:
  ModulesBuilder(const GlobalCompilationDatabase &CDB) : CDB(CDB) {}

  ModulesBuilder(const ModulesBuilder &) = delete;
  ModulesBuilder(ModulesBuilder &&) = delete;

  ModulesBuilder &operator=(const ModulesBuilder &) = delete;
  ModulesBuilder &operator=(ModulesBuilder &&) = delete;

  std::unique_ptr<PrerequisiteModules>
  buildPrerequisiteModulesFor(PathRef File, const ThreadsafeFS &TFS) const;

private:
  const GlobalCompilationDatabase &CDB;
};

} // namespace clangd
} // namespace clang

#endif
