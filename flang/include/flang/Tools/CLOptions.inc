//===-- CLOptions.inc -- command line options -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines some shared command-line options that can be used when
// debugging the test tools. This file must be included into the tool.

#define DisableOption(DOName, DOOption, DODescription) \
  static llvm::cl::opt<bool> disable##DOName("disable-" DOOption, \
      llvm::cl::desc("disable " DODescription " pass"), llvm::cl::init(false), \
      llvm::cl::Hidden)

namespace {
DisableOption(FirToLlvmIr, "fir-to-llvmir", "FIR to LLVM-IR dialect");
DisableOption(LlvmIrToLlvm, "llvm", "conversion to LLVM");

// Generic for adding a pass to the pass manager if it is not disabled.
template <typename F>
void addPassConditionally(
    mlir::PassManager &pm, llvm::cl::opt<bool> &disabled, F ctor) {
  if (!disabled)
    pm.addPass(ctor());
}

} // namespace

#if !defined(FLANG_EXCLUDE_CODEGEN)
namespace fir {

inline void addFIRToLLVMPass(mlir::PassManager &pm) {
  addPassConditionally(pm, disableFirToLlvmIr, fir::createFIRToLLVMPass);
}

inline void addLLVMDialectToLLVMPass(
    mlir::PassManager &pm, llvm::raw_ostream &output) {
  addPassConditionally(pm, disableLlvmIrToLlvm,
      [&]() { return fir::createLLVMDialectToLLVMPass(output); });
}

} // namespace fir
#undef FLANG_EXCLUDE_CODEGEN
#endif
