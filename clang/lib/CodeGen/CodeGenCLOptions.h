//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CODEGENCLOPTIONS_H
#define LLVM_CLANG_LIB_CODEGEN_CODEGENCLOPTIONS_H

#include "llvm/Support/PGOOptions.h"

#define OPTIONS_STRUCT_DECL
#include "CodeGenCLOptions.inc"

namespace llvm {
class LLVMContext;
} // namespace llvm

namespace clang::CodeGen {
const CodeGenCLOptions &getCodeGenCLOptions(const llvm::LLVMContext &Ctx);
} // namespace clang::CodeGen

#endif // LLVM_CLANG_LIB_CODEGEN_CODEGENCLOPTIONS_H
