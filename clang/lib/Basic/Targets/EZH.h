//===--- EZH.h - Declare EZH target feature support -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EZH TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_EZH_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_EZH_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY EZHTargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];
  bool HasBitsliceInterrupts = true;

public:
  EZHTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    resetDataLayout("e-m:e-p:32:32-i64:32-f64:32-n32-S32");
    RegParmMax = 4;
    MinGlobalAlign = 32;
    LongLongAlign = 32;
    DoubleAlign = 32;
    LongDoubleAlign = 32;
    SuitableAlign = 128;
  }


  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool isValidCPUName(StringRef Name) const override { return true; }

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override {}

  bool setCPU(const std::string &Name) override { return true; }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    for (const auto &Feature : Features) {
      if (Feature == "+bitslice-interrupts")
        HasBitsliceInterrupts = true;
      else if (Feature == "-bitslice-interrupts")
        HasBitsliceInterrupts = false;
    }
    return true;
  }

  bool hasFeature(StringRef Feature) const override {
    return Feature == "ezh" || (Feature == "bitslice-interrupts" && HasBitsliceInterrupts);
  }

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override {
    return {};
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return false;
  }

  std::string_view getClobbers() const override { return ""; }

  bool hasSjLjLowering() const override { return true; }
  bool hasBitIntType() const override { return true; }
  bool allowsLargerPreferedTypeAlignment() const override { return false; }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_EZH_H
