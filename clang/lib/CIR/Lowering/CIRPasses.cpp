//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements machinery for any CIR <-> CIR passes used by clang.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/TargetParser/Triple.h"

namespace cir {

/// Map a target triple to the ABI target that drives CallConvLowering.
/// Returns None for targets whose calling convention is not yet implemented.
static CallConvTarget getCallConvTarget(const llvm::Triple &triple) {
  if (triple.getArch() == llvm::Triple::x86_64)
    return CallConvTarget::X86_64;
  return CallConvTarget::None;
}

/// The AVX level the classifier uses to size a native vector, read from the
/// target ABI name.
static llvm::abi::X86AVXABILevel getX86AVXABILevel(llvm::StringRef abi) {
  if (abi == "avx512")
    return llvm::abi::X86AVXABILevel::AVX512;
  if (abi == "avx")
    return llvm::abi::X86AVXABILevel::AVX;
  return llvm::abi::X86AVXABILevel::None;
}

/// Whether `__attribute__((target(...)))` on a function may raise its AVX ABI
/// level above the command line's.  A target that opts out, and any ABI older
/// than the rule, stay at the module level.
static bool allowsX86TargetAttrAvx(const clang::ASTContext &astContext) {
  return !astContext.getTargetInfo().getTriple().isPS() &&
         astContext.getLangOpts().getClangABICompat() >
             clang::LangOptions::ClangABI::Ver23;
}

/// The x86_64 ABI-compatibility flags, derived from the target and the
/// requested compatibility version.  Every flag defaults to true in the ABI
/// library, which is not what any target computes: Clang11Compat is false for a
/// modern Linux target, so leaving it at the default classifies a union larger
/// than an eightbyte as though every member spanned its size.
static llvm::abi::ABICompatInfo
getX86ABICompatInfo(const clang::ASTContext &astContext) {
  const llvm::Triple &triple = astContext.getTargetInfo().getTriple();
  const clang::LangOptions &langOpts = astContext.getLangOpts();
  clang::LangOptions::ClangABI compat = langOpts.getClangABICompat();
  llvm::abi::ABICompatInfo abiCompat;
  abiCompat.HonorsRevision98 = !triple.isOSDarwin();
  abiCompat.ClassifyIntegerMMXAsSSE =
      compat > clang::LangOptions::ClangABI::Ver3_8 && !triple.isOSDarwin() &&
      !triple.isPS() && !triple.isOSFreeBSD();
  abiCompat.PassInt128VectorsInMem =
      compat > clang::LangOptions::ClangABI::Ver9 &&
      (triple.isOSLinux() || triple.isOSNetBSD());
  abiCompat.ReturnCXXRecordGreaterThan128InMem =
      compat > clang::LangOptions::ClangABI::Ver20 && !triple.isPS();
  abiCompat.Clang11Compat =
      compat <= clang::LangOptions::ClangABI::Ver11 || triple.isPS();
  return abiCompat;
}

mlir::LogicalResult
runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirContext,
                  clang::ASTContext &astContext, bool enableVerifier,
                  bool enableIdiomRecognizer, bool enableCIRSimplify,
                  bool enableLibOpt, llvm::StringRef libOptOptions,
                  bool enableCallConvLowering) {

  llvm::TimeTraceScope scope("CIR To CIR Passes");

  mlir::PassManager pm(&mlirContext);
  pm.addPass(mlir::createCIRCanonicalizePass());

  if (enableCIRSimplify)
    pm.addPass(mlir::createCIRSimplifyPass());

  if (enableIdiomRecognizer)
    pm.addPass(mlir::createIdiomRecognizerPass());

  if (enableLibOpt) {
    auto libOptPass = mlir::createLibOptPass();
    auto errorHandler = [](const llvm::Twine &) -> mlir::LogicalResult {
      return mlir::LogicalResult::failure();
    };

    if (libOptPass->initializeOptions(libOptOptions, errorHandler).failed())
      return mlir::failure();

    pm.addPass(std::move(libOptPass));
  }

  pm.addPass(mlir::createTargetLoweringPass());
  pm.addPass(mlir::createCXXABILoweringPass());

  if (enableCallConvLowering) {
    // CallConvLowering rewrites signatures and call sites using the classifier,
    // so it must run after CXXABILowering has lowered C++ ABI types to plain
    // records the classifier can handle.  Only the x86_64 System V classifier
    // is implemented; other targets are left unchanged.
    const clang::TargetInfo &targetInfo = astContext.getTargetInfo();
    CallConvTarget target = getCallConvTarget(targetInfo.getTriple());
    if (target != CallConvTarget::None)
      pm.addPass(mlir::createCallConvLoweringPass(
          target, getX86AVXABILevel(targetInfo.getABI()),
          allowsX86TargetAttrAvx(astContext), getX86ABICompatInfo(astContext)));
  }

  pm.addPass(mlir::createLoweringPreparePass(&astContext));

  pm.enableVerifier(enableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}

} // namespace cir

namespace mlir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createHoistAllocasPass());
  pm.addPass(createCIRFlattenCFGPass());
  pm.addPass(createCIREHABILoweringPass());
  pm.addPass(createGotoSolverPass());
}

} // namespace mlir
