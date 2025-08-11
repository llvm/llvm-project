//===- clang/unittests/AllClangUnitTests.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TargetSelect.h"

namespace {
struct RegisterAllLLVMTargets {
  RegisterAllLLVMTargets();
} gv;
} // namespace

// This dynamic initializer initializes all layers (TargetInfo, MC, CodeGen,
// AsmPrinter, etc) of all LLVM targets. This matches what cc1_main does on
// startup, and prevents tests from initializing some of the Target layers,
// which can interfere with tests that assume that lower target layers are
// registered if the TargetInfo is registered.
RegisterAllLLVMTargets::RegisterAllLLVMTargets() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
}
