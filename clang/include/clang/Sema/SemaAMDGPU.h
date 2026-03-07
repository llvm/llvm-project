//===----- SemaAMDGPU.h --- AMDGPU target-specific routines ---*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to AMDGPU.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAAMDGPU_H
#define LLVM_CLANG_SEMA_SEMAAMDGPU_H

#include "clang/AST/ASTFwd.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace clang {
class AttributeCommonInfo;
class Expr;
class ParsedAttr;

class SemaAMDGPU : public SemaBase {
  llvm::SmallPtrSet<Expr *, 32> ExpandedPredicates;
  llvm::SmallPtrSet<FunctionDecl *, 32> PotentiallyUnguardedBuiltinUsers;

public:
  SemaAMDGPU(Sema &S);

  bool CheckAMDGCNBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);

  /// Emits a diagnostic if the \p E is not an atomic ordering encoded in the C
  /// ABI format, or if the atomic ordering is not valid for the operation type
  /// as defined by \p MayLoad and \p MayStore. \returns true if a diagnostic
  /// was emitted.
  bool checkAtomicOrderingCABIArg(Expr *E, bool MayLoad, bool MayStore);

  bool checkCoopAtomicFunctionCall(CallExpr *TheCall, bool IsStore);
  bool checkAtomicMonitorLoad(CallExpr *TheCall);

  bool checkMovDPPFunctionCall(CallExpr *TheCall, unsigned NumArgs,
                               unsigned NumDataArgs);

  /// Create an AMDGPUWavesPerEUAttr attribute.
  AMDGPUFlatWorkGroupSizeAttr *
  CreateAMDGPUFlatWorkGroupSizeAttr(const AttributeCommonInfo &CI, Expr *Min,
                                    Expr *Max);

  /// addAMDGPUFlatWorkGroupSizeAttr - Adds an amdgpu_flat_work_group_size
  /// attribute to a particular declaration.
  void addAMDGPUFlatWorkGroupSizeAttr(Decl *D, const AttributeCommonInfo &CI,
                                      Expr *Min, Expr *Max);

  /// Create an AMDGPUWavesPerEUAttr attribute.
  AMDGPUWavesPerEUAttr *
  CreateAMDGPUWavesPerEUAttr(const AttributeCommonInfo &CI, Expr *Min,
                             Expr *Max);

  /// addAMDGPUWavePersEUAttr - Adds an amdgpu_waves_per_eu attribute to a
  /// particular declaration.
  void addAMDGPUWavesPerEUAttr(Decl *D, const AttributeCommonInfo &CI,
                               Expr *Min, Expr *Max);

  /// Create an AMDGPUMaxNumWorkGroupsAttr attribute.
  AMDGPUMaxNumWorkGroupsAttr *
  CreateAMDGPUMaxNumWorkGroupsAttr(const AttributeCommonInfo &CI, Expr *XExpr,
                                   Expr *YExpr, Expr *ZExpr);

  /// addAMDGPUMaxNumWorkGroupsAttr - Adds an amdgpu_max_num_work_groups
  /// attribute to a particular declaration.
  void addAMDGPUMaxNumWorkGroupsAttr(Decl *D, const AttributeCommonInfo &CI,
                                     Expr *XExpr, Expr *YExpr, Expr *ZExpr);

  void handleAMDGPUWavesPerEUAttr(Decl *D, const ParsedAttr &AL);
  void handleAMDGPUNumSGPRAttr(Decl *D, const ParsedAttr &AL);
  void handleAMDGPUNumVGPRAttr(Decl *D, const ParsedAttr &AL);
  void handleAMDGPUMaxNumWorkGroupsAttr(Decl *D, const ParsedAttr &AL);
  void handleAMDGPUFlatWorkGroupSizeAttr(Decl *D, const ParsedAttr &AL);

  /// Expand a valid use of the feature identification builtins into its
  /// corresponding sequence of instructions.
  Expr *ExpandAMDGPUPredicateBI(CallExpr *CE);
  bool IsPredicate(Expr *E) const;
  /// Diagnose unguarded usages of AMDGPU builtins and recommend guarding with
  /// __builtin_amdgcn_is_invocable
  void AddPotentiallyUnguardedBuiltinUser(FunctionDecl *FD);
  bool HasPotentiallyUnguardedBuiltinUsage(FunctionDecl *FD) const;
  void DiagnoseUnguardedBuiltinUsage(FunctionDecl *FD);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAAMDGPU_H
