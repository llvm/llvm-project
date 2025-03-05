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

#include "MitigationTagging.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

#include <string>
#include <vector>

namespace clang {
namespace CodeGen {

inline static std::string
MitigationKeyToString(enum MitigationKey key) noexcept {
  switch (key) {
  case MitigationKey::AUTO_VAR_INIT:
    return "auto-var-init";
  case MitigationKey::STACK_CLASH_PROTECTION:
    return "stack-clash-protection";
  case MitigationKey::STACK_PROTECTOR:
    return "stack-protector";
  case MitigationKey::STACK_PROTECTOR_STRONG:
    return "stack-protector-strong";
  case MitigationKey::STACK_PROTECTOR_ALL:
    return "stack-protector-all";
  case MitigationKey::CFI_VCALL:
    return "cfi-vcall";
  case MitigationKey::CFI_ICALL:
    return "cfi-icall";
  case MitigationKey::CFI_NVCALL:
    return "cfi-nvcall";
  }
}

void AttachMitigationMetadataToFunction(llvm::Function &F,
                                        enum MitigationKey key, bool enabled) {
  llvm::LLVMContext &Context = F.getContext();

  unsigned kindID = Context.getMDKindID("security_mitigations");

  llvm::Metadata *ValueMD = llvm::ConstantAsMetadata::get(
      llvm::ConstantInt::get(llvm::Type::getInt1Ty(Context), enabled));
  llvm::MDString *KeyMD =
      llvm::MDString::get(Context, MitigationKeyToString(key));

  llvm::MDNode *NewMD = llvm::MDNode::get(Context, {KeyMD, ValueMD});
  llvm::MDNode *ExistingMD = F.getMetadata(kindID);

  if (ExistingMD) {
    std::vector<llvm::Metadata *> MDs;
    for (unsigned i = 0, e = ExistingMD->getNumOperands(); i != e; ++i) {
      MDs.push_back(ExistingMD->getOperand(i));
    }
    MDs.push_back(NewMD);

    llvm::MDNode *CombinedMD = llvm::MDNode::get(Context, MDs);
    F.setMetadata(kindID, CombinedMD);
  } else {
    F.setMetadata(kindID, NewMD);
  }
}

void AttachMitigationMetadataToFunction(CodeGenFunction &CGF,
                                        enum MitigationKey key, bool enabled) {
  if (!CGF.CGM.getCodeGenOpts().MitigationAnalysis) {
    return;
  }
  AttachMitigationMetadataToFunction(*(CGF.CurFn), key, enabled);
}

} // namespace CodeGen
} // namespace clang
