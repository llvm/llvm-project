//===--- MitigationTagging.h - Emit LLVM Code from ASTs for a Module ------===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_MITIGATIONTAGGING_H
#define LLVM_CLANG_LIB_CODEGEN_MITIGATIONTAGGING_H

#include "CodeGenFunction.h"
#include "llvm/IR/Function.h"

namespace clang {
namespace CodeGen {

enum class MitigationKey {
  AUTO_VAR_INIT,

  STACK_CLASH_PROTECTION,

  STACK_PROTECTOR,
  STACK_PROTECTOR_STRONG,
  STACK_PROTECTOR_ALL,

  CFI_VCALL,
  CFI_ICALL,
  CFI_NVCALL,
};

void AttachMitigationMetadataToFunction(llvm::Function &F,
                                        enum MitigationKey key, bool enabled);
void AttachMitigationMetadataToFunction(CodeGenFunction &CGF,
                                        enum MitigationKey key, bool enabled);

} // namespace CodeGen
} // namespace clang

#endif
