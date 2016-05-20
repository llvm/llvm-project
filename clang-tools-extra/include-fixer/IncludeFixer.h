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

#include "SymbolIndexManager.h"
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
  /// \param SymbolIndexMgr A source for matching symbols to header files.
  /// \param Replacements Storage for the output of the fixer.
  /// \param StyleName Fallback style for reformatting.
  /// \param MinimizeIncludePaths whether inserted include paths are optimized.
  IncludeFixerActionFactory(
      SymbolIndexManager &SymbolIndexMgr, std::set<std::string> &Headers,
      std::vector<clang::tooling::Replacement> &Replacements,
      StringRef StyleName, bool MinimizeIncludePaths = true);

  ~IncludeFixerActionFactory() override;

  bool
  runInvocation(clang::CompilerInvocation *Invocation,
                clang::FileManager *Files,
                std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
                clang::DiagnosticConsumer *Diagnostics) override;

private:
  /// The client to use to find cross-references.
  SymbolIndexManager &SymbolIndexMgr;

  /// Headers to be added.
  std::set<std::string> &Headers;

  /// Replacements are written here.
  std::vector<clang::tooling::Replacement> &Replacements;

  /// Whether inserted include paths should be optimized.
  bool MinimizeIncludePaths;

  /// The fallback format style for formatting after insertion if no
  /// clang-format config file was found.
  std::string FallbackStyle;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H
