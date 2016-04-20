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

#include "XrefsDB.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"

namespace clang {
namespace include_fixer {

class IncludeFixerActionFactory : public clang::tooling::ToolAction {
public:
  /// \param Xrefs A source for matching symbols to header files.
  /// \param Replacements Storage for the output of the fixer.
  IncludeFixerActionFactory(
      XrefsDB &Xrefs, std::vector<clang::tooling::Replacement> &Replacements);
  ~IncludeFixerActionFactory();

  bool
  runInvocation(clang::CompilerInvocation *Invocation,
                clang::FileManager *Files,
                std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
                clang::DiagnosticConsumer *Diagnostics) override;

private:
  /// The client to use to find cross-references.
  XrefsDB &Xrefs;

  /// Replacements are written here.
  std::vector<clang::tooling::Replacement> &Replacements;
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_INCLUDEFIXER_H
