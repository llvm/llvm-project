//===----------------- PrerequisiteModules.h ---------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREREQUISITEMODULES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PREREQUISITEMODULES_H

#include "Compiler.h"
#include "support/Path.h"

#include "clang/Lex/HeaderSearchOptions.h"

#include "llvm/ADT/StringSet.h"

namespace clang {
namespace clangd {

class ModulesBuilder;

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
///   import a;
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

  /// Return true if the modile file specified by ModuleName is built.
  /// Note that this interface will only check the existence of the module
  /// file instead of checking the validness of the module file.
  virtual bool isModuleUnitBuilt(StringRef ModuleName) const = 0;

  virtual ~PrerequisiteModules() = default;

private:
  friend class ModulesBuilder;

  /// Add a module file to the PrerequisiteModules.
  virtual void addModuleFile(StringRef ModuleName,
                             StringRef ModuleFilePath) = 0;
};

} // namespace clangd
} // namespace clang

#endif
