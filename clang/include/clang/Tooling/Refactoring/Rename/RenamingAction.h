//===--- RenamingAction.h - Clang refactoring library ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides an action to rename every symbol at a point.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_RENAME_RENAMING_ACTION_H
#define LLVM_CLANG_TOOLING_REFACTOR_RENAME_RENAMING_ACTION_H

#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Refactoring/RefactoringOptions.h"
#include "clang/Tooling/Refactoring/Rename/SymbolOccurrences.h"
#include "llvm/Support/Error.h"

namespace clang {
class ASTConsumer;
class CompilerInstance;

namespace tooling {

class RenamingAction {
public:
  RenamingAction(const std::vector<std::string> &NewNames,
                 const std::vector<std::string> &PrevNames,
                 const std::vector<std::vector<std::string>> &USRList,
                 std::map<std::string, tooling::Replacements> &FileToReplaces,
                 bool PrintLocations = false)
      : NewNames(NewNames), PrevNames(PrevNames), USRList(USRList),
        FileToReplaces(FileToReplaces), PrintLocations(PrintLocations) {}

  std::unique_ptr<ASTConsumer> newASTConsumer();

private:
  const std::vector<std::string> &NewNames, &PrevNames;
  const std::vector<std::vector<std::string>> &USRList;
  std::map<std::string, tooling::Replacements> &FileToReplaces;
  bool PrintLocations;
};

class NewNameOption : public RequiredRefactoringOption<std::string> {
public:
  StringRef getName() const override { return "new-name"; }
  StringRef getDescription() const override {
    return "The new name to change the symbol to";
  }
};

/// Returns source replacements that correspond to the rename of the given
/// symbol occurrences.
llvm::Expected<std::vector<AtomicChange>>
createRenameReplacements(const SymbolOccurrences &Occurrences,
                         const SourceManager &SM, const SymbolName &NewName);

/// Rename all symbols identified by the given USRs.
class QualifiedRenamingAction {
public:
  QualifiedRenamingAction(
      const std::vector<std::string> &NewNames,
      const std::vector<std::vector<std::string>> &USRList,
      std::map<std::string, tooling::Replacements> &FileToReplaces)
      : NewNames(NewNames), USRList(USRList), FileToReplaces(FileToReplaces) {}

  std::unique_ptr<ASTConsumer> newASTConsumer();

private:
  /// New symbol names.
  const std::vector<std::string> &NewNames;

  /// A list of USRs. Each element represents USRs of a symbol being renamed.
  const std::vector<std::vector<std::string>> &USRList;

  /// A file path to replacements map.
  std::map<std::string, tooling::Replacements> &FileToReplaces;
};

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_RENAME_RENAMING_ACTION_H
