//===- CIRABIClassifierX86_64.cpp - x86_64 System V classification --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the x86_64 System V classifier, which classifies with
// the LLVM ABI Lowering Library's SysV x86_64 target.
//
//===----------------------------------------------------------------------===//

#include "CIRABIClassifier.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include <array>

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace {

class X86_64ABIClassifier final : public CIRABIClassifierCommon {
public:
  X86_64ABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp,
                      const ABIClassifierOptions &options)
      : CIRABIClassifierCommon(dl, modOp), baseAvxLevel(options.x86AvxAbiLevel),
        allowsTargetAttrAvx(options.allowsX86TargetAttrAvx),
        abiCompat(options.x86AbiCompat) {}

  /// cir.va_arg is expanded into the SysV register-save-area / overflow-area
  /// sequence, which reads the fetched type's classification as an unnamed
  /// argument, as classic's X86_64ABIInfo::EmitVAArg does.
  std::optional<ArgClassification> classifyVAArg(cir::VAArgOp vaArg) override {
    return classifyVarArgType(
        vaArg.getType(), getTargetInfo(vaArg->getParentOfType<cir::FuncOp>()),
        [&]() { return vaArg->emitOpError(); });
  }

protected:
  /// The AVX level starts at the module's and may be raised by
  /// \p featureSource's recorded target features.
  const llvm::abi::TargetInfo &
  getTargetInfo(cir::FuncOp featureSource) override;

private:
  /// One library classifier per AVX level, built on first use.  A
  /// per-function target attribute can raise the level above the module's.
  static constexpr unsigned numAvxLevels =
      static_cast<unsigned>(llvm::abi::X86AVXABILevel::Last) + 1;
  std::array<std::unique_ptr<llvm::abi::TargetInfo>, numAvxLevels> targets;

  llvm::abi::X86AVXABILevel baseAvxLevel;
  bool allowsTargetAttrAvx;

  /// The x86_64 flags whose value depends on the target and the requested ABI
  /// compatibility version.
  llvm::abi::X86ABICompatInfo abiCompat;
};

const llvm::abi::TargetInfo &
X86_64ABIClassifier::getTargetInfo(cir::FuncOp featureSource) {
  llvm::abi::X86AVXABILevel level = baseAvxLevel;
  assert(featureSource && "function required for getTargetInfo");
  if (allowsTargetAttrAvx)
    level = funcAvxLevel(featureSource, baseAvxLevel);
  assert(static_cast<unsigned>(level) < numAvxLevels &&
         "a new X86AVXABILevel must move X86AVXABILevel::Last");
  std::unique_ptr<llvm::abi::TargetInfo> &slot =
      targets[static_cast<unsigned>(level)];
  if (!slot)
    slot =
        llvm::abi::createX86_64TargetInfo(typeMapper.getTypeBuilder(), level,
                                          /*Has64BitPointers=*/true, abiCompat);
  return *slot;
}

} // namespace

std::unique_ptr<CIRABIClassifier>
cir::createX86_64ABIClassifier(const mlir::DataLayout &dl, mlir::ModuleOp modOp,
                               const ABIClassifierOptions &options) {
  return std::make_unique<X86_64ABIClassifier>(dl, modOp, options);
}
