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

llvm::abi::AArch64ABIOptions cir::getAArch64ABIOptions(mlir::ModuleOp modOp) {
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

namespace {

class AArch64ABIClassifier final : public CIRABIClassifierCommon {
public:
  AArch64ABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp,
                       const llvm::abi::AArch64ABIOptions &opts)
      : CIRABIClassifierCommon(dl, modOp), abiKind(opts.Kind),
        targetInfo(llvm::abi::createAArch64TargetInfo(
            typeMapper.getTypeBuilder(), opts)) {}

  /// Classify the type fetched by \p vaArg as an unnamed argument with a full
  /// register budget.
  mlir::FailureOr<ArgClassification>
  classifyVAArg(cir::VAArgOp vaArg) override {
    switch (abiKind) {
    case llvm::abi::AArch64ABIKind::DarwinPCS:
      vaArg->emitOpError()
          << "not yet implemented in CallConvLowering for Darwin AArch64";
      return mlir::failure();
    case llvm::abi::AArch64ABIKind::Win64:
      vaArg->emitOpError()
          << "not yet implemented in CallConvLowering for Windows AArch64";
      return mlir::failure();
    case llvm::abi::AArch64ABIKind::AAPCS:
    case llvm::abi::AArch64ABIKind::AAPCSSoft:
      break;
    }
    return classifyVarArgType(vaArg.getType(), *targetInfo,
                              [&]() { return vaArg->emitOpError(); });
  }

protected:
  /// AArch64 has no per-function features that change the classification,
  /// so every function shares the module's classifier.
  const llvm::abi::TargetInfo &getTargetInfo(cir::FuncOp) override {
    return *targetInfo;
  }

private:
  llvm::abi::AArch64ABIKind abiKind;
  std::unique_ptr<llvm::abi::TargetInfo> targetInfo;
};

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createAArch64ABIClassifier(const mlir::DataLayout &dl,
                                mlir::ModuleOp modOp,
                                const llvm::abi::AArch64ABIOptions &opts) {
  return std::make_unique<AArch64ABIClassifier>(dl, modOp, opts);
}
