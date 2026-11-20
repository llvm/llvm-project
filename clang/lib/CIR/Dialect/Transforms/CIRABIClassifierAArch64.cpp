//===- CIRABIClassifierAArch64.cpp - AArch64 classification ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64 classifier, which classifies with the LLVM
// ABI Lowering Library's AArch64 target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetCXXABI.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/TargetParser/Triple.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace {

/// The AArch64 options the library classifies with, derived from \p modOp's
/// attributes the way classic CodeGen derives them from the target and the
/// language options.  A module with no triple is classified as AAPCS64
/// little-endian, and one with no source language as C.
llvm::abi::AArch64ABIOptions getAArch64ABIOptions(mlir::ModuleOp modOp) {
  llvm::Triple triple("aarch64-unknown-linux-gnu");
  if (auto tripleAttr = modOp->getAttrOfType<mlir::StringAttr>(
          cir::CIRDialect::getTripleAttrName()))
    triple = llvm::Triple(tripleAttr.getValue());

  llvm::StringRef abi;
  if (auto abiAttr = modOp->getAttrOfType<mlir::StringAttr>(
          cir::CIRDialect::getTargetABIAttrName()))
    abi = abiAttr.getValue();

  llvm::abi::AArch64ABIOptions opts;
  if (abi == "darwinpcs")
    opts.Kind = llvm::abi::AArch64ABIKind::DarwinPCS;
  else if (triple.isOSWindows())
    opts.Kind = llvm::abi::AArch64ABIKind::Win64;
  else if (abi == "aapcs-soft")
    opts.Kind = llvm::abi::AArch64ABIKind::AAPCSSoft;
  else
    opts.Kind = llvm::abi::AArch64ABIKind::AAPCS;

  auto lang = modOp->getAttrOfType<cir::SourceLanguageAttr>(
      cir::CIRDialect::getSourceLanguageAttrName());
  opts.IsCXX = lang && (lang.isCXX() || lang.isOpenCLCXX());

  opts.IsILP32 = triple.getArch() == llvm::Triple::aarch64_32;
  opts.IsMachO = triple.isOSBinFormatMachO();
  opts.IsAndroidOrOHOS = triple.isAndroid() || triple.isOHOSFamily();
  opts.IsWindowsArm64EC = triple.isWindowsArm64EC();
  if (auto cxxABIAttr = modOp->getAttrOfType<mlir::StringAttr>(
          cir::CIRDialect::getCXXABIAttrName()))
    opts.IsMicrosoftCXXABI =
        clang::TargetCXXABI::getKind(cxxABIAttr.getValue()) ==
        clang::TargetCXXABI::Microsoft;
  else
    opts.IsMicrosoftCXXABI = triple.isWindowsMSVCEnvironment();
  opts.IsBigEndian = !triple.isLittleEndian();

  auto compat = clang::LangOptions::ClangABI::Latest;
  if (auto loweringOpts = modOp->getAttrOfType<cir::LoweringOptionsAttr>(
          cir::CIRDialect::getLoweringOptionsAttrName()))
    compat = static_cast<clang::LangOptions::ClangABI>(
        loweringOpts.getClangAbiCompat());
  opts.CompatInfo.IsMatrixHA = compat > clang::LangOptions::ClangABI::Ver23;
  return opts;
}

class AArch64ABIClassifier final : public CIRABIClassifierCommon {
public:
  AArch64ABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp)
      : CIRABIClassifierCommon(dl, modOp),
        targetInfo(llvm::abi::createAArch64TargetInfo(
            typeMapper.getTypeBuilder(), getAArch64ABIOptions(modOp))) {}

  /// The AArch64 `va_list` and the expansion of `cir.va_arg` through it are
  /// not implemented yet.
  mlir::FailureOr<ArgClassification>
  classifyVAArg(cir::VAArgOp vaArg) override {
    vaArg->emitOpError()
        << "not yet implemented in CallConvLowering for AArch64";
    return mlir::failure();
  }

protected:
  /// AArch64 has no per-function features that change the classification,
  /// so every function shares the module's classifier.
  const llvm::abi::TargetInfo &getTargetInfo(cir::FuncOp) override {
    return *targetInfo;
  }

private:
  std::unique_ptr<llvm::abi::TargetInfo> targetInfo;
};

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createAArch64ABIClassifier(const mlir::DataLayout &dl,
                                mlir::ModuleOp modOp) {
  return std::make_unique<AArch64ABIClassifier>(dl, modOp);
}
