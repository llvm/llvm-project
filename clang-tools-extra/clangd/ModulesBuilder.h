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
// FIXME: Supporting reusing module files across different versions and
// different source files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULES_BUILDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_MODULES_BUILDER_H

#include "GlobalCompilationDatabase.h"
#include "ProjectModules.h"

#include "support/Path.h"
#include "support/ThreadsafeFS.h"

#include "llvm/ADT/SmallString.h"

#include <memory>

namespace clang {
namespace clangd {

class PrerequisiteModules;

/// This class handles building module files for a given source file.
///
/// In the future, we want the class to manage the module files acorss
/// different versions and different source files.
class ModulesBuilder {
public:
  enum class ModulesBuilderKind {
    StandaloneModulesBuilder,
    ReusableModulesBuilder
  };

  static std::unique_ptr<ModulesBuilder>
  create(ModulesBuilderKind Kind, const GlobalCompilationDatabase &CDB);

  virtual ~ModulesBuilder() = default;

  virtual std::unique_ptr<PrerequisiteModules>
  buildPrerequisiteModulesFor(PathRef File, const ThreadsafeFS *TFS) = 0;
};

} // namespace clangd
} // namespace clang

#endif
