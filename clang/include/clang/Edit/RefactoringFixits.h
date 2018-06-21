//===--- RefactoringFixits.h - Fixit producers for refactorings -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EDIT_REFACTORING_FIXITS_H
#define LLVM_CLANG_EDIT_REFACTORING_FIXITS_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {

class ASTContext;
class SwitchStmt;
class EnumDecl;
class ObjCContainerDecl;

namespace edit {

/**
 * Generates the fix-its that perform the "add missing switch cases" refactoring
 * operation.
 */
void fillInMissingSwitchEnumCases(
    ASTContext &Context, const SwitchStmt *Switch, const EnumDecl *Enum,
    const DeclContext *SwitchContext,
    llvm::function_ref<void(const FixItHint &)> Consumer);

/// Responsible for the fix-its that perform the
/// "add missing protocol requirements" refactoring operation.
namespace fillInMissingProtocolStubs {

class FillInMissingProtocolStubsImpl;
class FillInMissingProtocolStubs {
  std::unique_ptr<FillInMissingProtocolStubsImpl> Impl;

public:
  FillInMissingProtocolStubs();
  ~FillInMissingProtocolStubs();
  FillInMissingProtocolStubs(FillInMissingProtocolStubs &&);
  FillInMissingProtocolStubs &operator=(FillInMissingProtocolStubs &&);

  /// Initiate the FillInMissingProtocolStubs edit.
  ///
  /// \returns true on Error.
  bool initiate(ASTContext &Context, const ObjCContainerDecl *Container);
  bool hasMissingRequiredMethodStubs();
  void perform(ASTContext &Context,
               llvm::function_ref<void(const FixItHint &)> Consumer);
};

void addMissingProtocolStubs(
    ASTContext &Context, const ObjCContainerDecl *Container,
    llvm::function_ref<void(const FixItHint &)> Consumer);

} // end namespace fillInMissingProtocolStubs

} // end namespace edit
} // end namespace clang

#endif // LLVM_CLANG_EDIT_REFACTORING_FIXITS_H
