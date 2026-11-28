//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenCLOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Option/LibraryOptions.h"

#define OPTIONS_STRUCT_DEFS
#include "CodeGenCLOptions.inc"

LLVM_REGISTER_LIBRARY_OPTIONS(clang::CodeGen::CodeGenCLOptions)

const clang::CodeGen::CodeGenCLOptions &
clang::CodeGen::getCodeGenCLOptions(const llvm::LLVMContext &Ctx) {
  return Ctx.getLibraryOptions().get<CodeGenCLOptions>();
}
