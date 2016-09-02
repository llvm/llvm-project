//===-- tools/extra/clang-reorder-fields/ReorderFieldsAction.h -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the ReorderFieldsAction class and
/// the FieldPosition struct.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_REORDER_FIELDS_ACTION_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_REORDER_FIELDS_ACTION_H

#include "clang/Tooling/Refactoring.h"

namespace clang {
class ASTConsumer;

namespace reorder_fields {

class ReorderFieldsAction {
  llvm::StringRef RecordName;
  llvm::ArrayRef<std::string> DesiredFieldsOrder;
  std::map<std::string, tooling::Replacements> &Replacements;

public:
  ReorderFieldsAction(
      llvm::StringRef RecordName,
      llvm::ArrayRef<std::string> DesiredFieldsOrder,
      std::map<std::string, tooling::Replacements> &Replacements)
      : RecordName(RecordName), DesiredFieldsOrder(DesiredFieldsOrder),
        Replacements(Replacements) {}

  ReorderFieldsAction(const ReorderFieldsAction &) = delete;
  ReorderFieldsAction &operator=(const ReorderFieldsAction &) = delete;

  std::unique_ptr<ASTConsumer> newASTConsumer();
};
} // namespace reorder_fields
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_REORDER_FIELDS_ACTION_H
