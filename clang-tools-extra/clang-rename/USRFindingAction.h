//===--- tools/extra/clang-rename/USRFindingAction.h - Clang rename tool --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides an action to find all relevant USRs at a point.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDING_ACTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDING_ACTION_H

#include "clang/Frontend/FrontendAction.h"

namespace clang {
class ASTConsumer;
class CompilerInstance;
class NamedDecl;

namespace rename {

struct USRFindingAction {
  USRFindingAction(unsigned Offset, const std::string &Name)
      : SymbolOffset(Offset), OldName(Name) {}
  std::unique_ptr<ASTConsumer> newASTConsumer();

  // \brief get the spelling of the USR(s) as it would appear in source files.
  const std::string &getUSRSpelling() { return SpellingName; }

  const std::vector<std::string> &getUSRs() { return USRs; }

private:
  unsigned SymbolOffset;
  std::string OldName;
  std::string SpellingName;
  std::vector<std::string> USRs;
};

} // namespace rename
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDING_ACTION_H
