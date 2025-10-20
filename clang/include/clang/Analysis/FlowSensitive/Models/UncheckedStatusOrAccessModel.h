//===- UncheckedStatusOrAccessModel.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDSTATUSORACCESSMODEL_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDSTATUSORACCESSMODEL_H

#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace clang::dataflow::statusor_model {

// The helper functions exported here are for use of downstream vendor
// extensions of this model.

// Match declaration of `absl::StatusOr<T>` and bind `T` to "T".
clang::ast_matchers::DeclarationMatcher statusOrClass();
// Match declaration of `absl::Status`.
clang::ast_matchers::DeclarationMatcher statusClass();
// Match declaration of `absl::internal_statusor::OperatorBase`.
clang::ast_matchers::DeclarationMatcher statusOrOperatorBaseClass();
clang::ast_matchers::TypeMatcher statusOrType();

// Get RecordStorageLocation for the `Status` contained in the `StatusOr`
RecordStorageLocation &locForStatus(RecordStorageLocation &StatusOrLoc);
// Get the StorageLocation for the OK boolean in the `Status`
StorageLocation &locForOk(RecordStorageLocation &StatusLoc);
// Get the OK boolean in the `Status`, and initialize it if necessary.
BoolValue &valForOk(RecordStorageLocation &StatusLoc, Environment &Env);
// Get synthetic fields for the types modelled by
// `UncheckedStatusOrAccessModel`.
llvm::StringMap<QualType> getSyntheticFields(QualType Ty, QualType StatusType,
                                             const CXXRecordDecl &RD);

// Initialize the synthetic fields of the `StatusOr`.
// N.B. if it is already initialized, the value gets reset.
BoolValue &initializeStatusOr(RecordStorageLocation &StatusOrLoc,
                              Environment &Env);
// Initialize the synthetic fields of the `Status`.
// N.B. if it is already initialized, the value gets reset.
BoolValue &initializeStatus(RecordStorageLocation &StatusLoc, Environment &Env);

bool isRecordTypeWithName(QualType Type, llvm::StringRef TypeName);
// Return true if `Type` is instantiation of `absl::StatusOr<T>`
bool isStatusOrType(QualType Type);
// Return true if `Type` is `absl::Status`
bool isStatusType(QualType Type);

// Get `QualType` for `absl::Status`, or default-constructed
// QualType if it does not exist.
QualType findStatusType(const ASTContext &Ctx);

struct UncheckedStatusOrAccessModelOptions {};

// Dataflow analysis that discovers unsafe uses of StatusOr values.
class UncheckedStatusOrAccessModel
    : public DataflowAnalysis<UncheckedStatusOrAccessModel, NoopLattice> {
public:
  explicit UncheckedStatusOrAccessModel(ASTContext &Ctx, Environment &Env);

  static Lattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, Lattice &L, Environment &Env);

private:
  CFGMatchSwitch<TransferState<Lattice>> TransferMatchSwitch;
};

using LatticeTransferState =
    TransferState<UncheckedStatusOrAccessModel::Lattice>;

// Extend the Builder with the transfer functions for
// `UncheckedStatusOrAccessModel`. This is useful to write downstream models
// that extend the model.
CFGMatchSwitch<LatticeTransferState>
buildTransferMatchSwitch(ASTContext &Ctx,
                         CFGMatchSwitchBuilder<LatticeTransferState> Builder);

class UncheckedStatusOrAccessDiagnoser {
public:
  explicit UncheckedStatusOrAccessDiagnoser(
      UncheckedStatusOrAccessModelOptions Options = {});

  llvm::SmallVector<SourceLocation> operator()(
      const CFGElement &Elt, ASTContext &Ctx,
      const TransferStateForDiagnostics<UncheckedStatusOrAccessModel::Lattice>
          &State);

private:
  CFGMatchSwitch<const Environment, llvm::SmallVector<SourceLocation>>
      DiagnoseMatchSwitch;
};

} // namespace clang::dataflow::statusor_model

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_UNCHECKEDSTATUSORACCESSMODEL_H
