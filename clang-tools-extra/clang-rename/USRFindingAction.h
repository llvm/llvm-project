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

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

#include <string>
#include <vector>

namespace clang {
class ASTConsumer;
class CompilerInstance;
class NamedDecl;

namespace rename {

struct USRFindingAction {
  USRFindingAction(ArrayRef<unsigned> SymbolOffsets,
                   ArrayRef<std::string> QualifiedNames)
      : SymbolOffsets(SymbolOffsets), QualifiedNames(QualifiedNames),
        ErrorOccurred(false) {}
  std::unique_ptr<ASTConsumer> newASTConsumer();

  ArrayRef<std::string> getUSRSpellings() { return SpellingNames; }
  ArrayRef<std::vector<std::string>> getUSRList() { return USRList; }
  bool errorOccurred() { return ErrorOccurred; }

private:
  std::vector<unsigned> SymbolOffsets;
  std::vector<std::string> QualifiedNames;
  std::vector<std::string> SpellingNames;
  std::vector<std::vector<std::string>> USRList;
  bool ErrorOccurred;
};

} // namespace rename
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_FINDING_ACTION_H
