//===--- SuperH.h - Declare SuperH target feature support -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares SuperH TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_SUPERH_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_SUPERH_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

// SuperH Target
class LLVM_LIBRARY_VISIBILITY SuperHTargetInfo : public TargetInfo {
public:
  SuperHTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    TLSSupported = false;
    PointerWidth = 32;
    PointerAlign = 32;
    ShortWidth = 16;
    ShortAlign = 32;
    IntWidth = 32;
    IntAlign = 32;
    LongWidth = 32;
    LongAlign = 32;
    LongLongWidth = 64;
    LongLongAlign = 64;
    SuitableAlign = 32;
    DefaultAlignForAttributeAligned = 32;
    HalfWidth = 16;
    HalfAlign = 32;
    FloatWidth = 32;
    FloatAlign = 32;
    DoubleWidth = 32;
    DoubleAlign = 32;
    DoubleFormat = &llvm::APFloat::IEEEsingle();
    LongDoubleWidth = 64;
    LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    Char16Type = UnsignedShort;
    WIntType = SignedInt;
    Int16Type = SignedShort;
    Char32Type = UnsignedLong;
    SigAtomicType = SignedChar;
    resetDataLayout();
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override {
    return {};
  }

  bool allowsLargerPreferedTypeAlignment() const override { return false; }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  std::string_view getClobbers() const override { return ""; }

  ArrayRef<const char *> getGCCRegNames() const override {
    static const char *const GCCRegNames[] = {
        "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",  "r8",
        "r9",  "r10", "r11", "r12", "r13", "r14", "r15"
    };
    return llvm::ArrayRef(GCCRegNames);
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return {};
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    return false;
  }

  bool isValidCPUName(StringRef Name) const override;
  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;
  bool setCPU(const std::string &Name) override;
  std::optional<std::string> handleAsmEscapedChar(char EscChar) const override;
  StringRef getABI() const override { return ABI; }

  std::pair<unsigned, unsigned> hardwareInterferenceSizes() const override {
    return std::make_pair(32, 32);
  }

protected:
  std::string CPU;
  StringRef ABI;
};

}
}

#endif