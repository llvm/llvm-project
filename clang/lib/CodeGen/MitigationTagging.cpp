//===--- MitigationTagging.cpp - Emit LLVM Code from ASTs for a Module ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This enables tagging functions with metadata to indicate mitigations are
// applied to them.
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/MitigationTagging.h"
#include "CodeGenFunction.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace CodeGen {

///
/// Store metadata (tied to the function) related to enablement of mitigations.
/// @param Key - identifier for the mitigation
/// @param MitigationEnable - if the mitigation is enabled for the target
/// function
///
void AttachMitigationMetadataToFunction(llvm::Function &F,
                                        enum llvm::MitigationKey Key,
                                        bool MitigationEnable) {
  llvm::LLVMContext &Context = F.getContext();

  const auto &MitigationToString = llvm::GetMitigationMetadataMapping();
  auto KV = MitigationToString.find(Key);
  if (KV == MitigationToString.end())
    return;

  unsigned KindID = Context.getMDKindID(KV->second);

  // Only set once
  if (!F.getMetadata(KindID)) {
    llvm::Metadata *ValueMD =
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::Type::getInt1Ty(Context), MitigationEnable));
    F.setMetadata(KindID, llvm::MDNode::get(Context, ValueMD));
  }
}

///
/// Store metadata (tied to the function) related to enablement of mitigations.
/// Checks if MitigationAnalysis CodeGenOpt is set first and is a no-op if
/// unset.
/// @param Key - identifier for the mitigation
/// @param MitigationEnable - if the mitigation is enabled for the target
/// function
///
void AttachMitigationMetadataToFunction(CodeGenFunction &CGF,
                                        enum llvm::MitigationKey Key,
                                        bool MitigationEnable) {
  if (CGF.CGM.getCodeGenOpts().MitigationAnalysis)
    AttachMitigationMetadataToFunction(*(CGF.CurFn), Key, MitigationEnable);
}

} // namespace CodeGen
} // namespace clang
