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

#include "llvm/IR/Function.h"
#include "llvm/Support/MitigationMarker.h"

namespace clang {
namespace CodeGen {

class CodeGenFunction;

void AttachMitigationMetadataToFunction(llvm::Function &F,
                                        enum llvm::MitigationKey Key,
                                        bool MitigationEnable);
void AttachMitigationMetadataToFunction(CodeGenFunction &CGF,
                                        enum llvm::MitigationKey Key,
                                        bool MitigationEnable);

} // namespace CodeGen
} // namespace clang

#endif
