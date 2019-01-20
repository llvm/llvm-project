//===--- SanitizerMetadata.h - Metadata for sanitizers ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Class which emits metadata consumed by sanitizer instrumentation passes.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LIB_CODEGEN_SANITIZERMETADATA_H
#define LLVM_CLANG_LIB_CODEGEN_SANITIZERMETADATA_H

#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"

namespace llvm {
class GlobalVariable;
class Instruction;
class MDNode;
}

namespace clang {
class VarDecl;

namespace CodeGen {

class CodeGenModule;

class SanitizerMetadata {
  SanitizerMetadata(const SanitizerMetadata &) = delete;
  void operator=(const SanitizerMetadata &) = delete;

  CodeGenModule &CGM;
public:
  SanitizerMetadata(CodeGenModule &CGM);
  void reportGlobalToASan(llvm::GlobalVariable *GV, const VarDecl &D,
                          bool IsDynInit = false);
  void reportGlobalToASan(llvm::GlobalVariable *GV, SourceLocation Loc,
                          StringRef Name, QualType Ty, bool IsDynInit = false,
                          bool IsBlacklisted = false);
  void disableSanitizerForGlobal(llvm::GlobalVariable *GV);
  void disableSanitizerForInstruction(llvm::Instruction *I);
private:
  llvm::MDNode *getLocationMetadata(SourceLocation Loc);
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
