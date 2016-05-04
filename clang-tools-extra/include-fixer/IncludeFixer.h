//===-- IncludeFixer.h - Include inserter -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H

#include "XrefsDBManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include <memory>
#include <vector>

namespace clang {

class CompilerInvocation;
class DiagnosticConsumer;
class FileManager;
class PCHContainerOperations;

namespace include_fixer {

class IncludeFixerActionFactory : public clang::tooling::ToolAction {
public:
  /// \param XrefsDBMgr A source for matching symbols to header files.
  /// \param Replacements Storage for the output of the fixer.
  /// \param MinimizeIncludePaths whether inserted include paths are optimized.
  IncludeFixerActionFactory(
      XrefsDBManager &XrefsDBMgr,
      std::vector<clang::tooling::Replacement> &Replacements,
      bool MinimizeIncludePaths = true);
  ~IncludeFixerActionFactory() override;

  bool
  runInvocation(clang::CompilerInvocation *Invocation,
                clang::FileManager *Files,
                std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
                clang::DiagnosticConsumer *Diagnostics) override;

private:
  /// The client to use to find cross-references.
  XrefsDBManager &XrefsDBMgr;

  /// Replacements are written here.
  std::vector<clang::tooling::Replacement> &Replacements;

  /// Whether inserted include paths should be optimized.
  bool MinimizeIncludePaths;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H
