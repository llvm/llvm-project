//===--- Sparc.h - declare sparc target feature support ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Sparc TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_SPARC_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_SPARC_H
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/SparcTargetParser.h"
#include "llvm/TargetParser/Triple.h"
namespace clang {
namespace targets {
// Shared base class for SPARC v8 (32-bit) and SPARC v9 (64-bit).
class LLVM_LIBRARY_VISIBILITY SparcTargetInfo : public TargetInfo {
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char *const GCCRegNames[];
  bool SoftFloat;

protected:
  // Features of the CPU and target features, as defined by the backend.
  bool IsV9 = false;
  bool IsLeon = false;
  bool HasLeonCASA = false;
  bool SoftMulDiv = false;

public:
  SparcTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple), SoftFloat(false) {
    HasStrictFP = true;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0)
      return 24;
    if (RegNo == 1)
      return 25;
    return -1;
  }

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override;

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool hasFeature(StringRef Feature) const override;

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override {
    // FIXME: Implement!
    return {};
  }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    // FIXME: Implement!
    switch (*Name) {
    case 'I': // Signed 13-bit constant
    case 'J': // Zero
    case 'K': // 32-bit constant with the low 12 bits clear
    case 'L': // A constant in the range supported by movcc (11-bit signed imm)
    case 'M': // A constant in the range supported by movrcc (19-bit signed imm)
    case 'N': // Same as 'K' but zext (required for SIMode)
    case 'O': // The constant 4096
      return true;

    case 'f':
    case 'e':
      info.setAllowsRegister();
      return true;
    }
    return false;
  }
  std::string_view getClobbers() const override { return "~{y}"; }

  bool isValidCPUName(StringRef Name) const override;

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;

  bool setCPU(StringRef Name) override { return isValidCPUName(Name); }

  std::pair<unsigned, unsigned> hardwareInterferenceSizes() const override {
    return std::make_pair(32, 32);
  }
};

// SPARC v8 is the 32-bit mode selected by Triple::sparc.
class LLVM_LIBRARY_VISIBILITY SparcV8TargetInfo : public SparcTargetInfo {
  bool V8Plus = false;

public:
  SparcV8TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SparcTargetInfo(Triple, Opts) {
    resetDataLayout();
    // NetBSD / OpenBSD use long (same as llvm default); everyone else uses int.
    switch (getTriple().getOS()) {
    default:
      SizeType = UnsignedInt;
      IntPtrType = SignedInt;
      PtrDiffType = SignedInt;
      break;
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
      SizeType = UnsignedLong;
      IntPtrType = SignedLong;
      PtrDiffType = SignedLong;
      break;
    }

    // Base long double format on the triple.
    if (getTriple().getDefaultLongDoubleFormat() ==
        llvm::LongDoubleFormat::IEEEdouble) {
      LongDoubleWidth = 64;
      LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    } else {
      LongDoubleWidth = 128;
      LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEquad();
    }

    // We're willing to do atomic ops on up to 64 bits. Which of those are
    // lock-free is decided in setMaxAtomicWidth.
    MaxAtomicPromoteWidth = 64;
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    if (!SparcTargetInfo::handleTargetFeatures(Features, Diags))
      return false;
    V8Plus = llvm::is_contained(Features, "+v8plus");
    return true;
  }

  void setMaxAtomicWidth() override {
    if (IsV9)
      // Like GCC, 64-bit atomics in 32-bit code need V8+ mode, where the
      // %g and %o registers are 64 bits wide.
      MaxAtomicInlineWidth = V8Plus ? 64 : 32;
    else
      // Plain V8 lacks CAS; only the LEON and Myriad 2 CPUs with CASA have it.
      MaxAtomicInlineWidth = HasLeonCASA ? 32 : 0;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool hasBitIntType() const override { return true; }
};

// SPARCV8el is the 32-bit little-endian mode selected by Triple::sparcel.
class LLVM_LIBRARY_VISIBILITY SparcV8elTargetInfo : public SparcV8TargetInfo {
public:
  SparcV8elTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SparcV8TargetInfo(Triple, Opts) {
    resetDataLayout();
  }
};

// SPARC v9 is the 64-bit mode selected by Triple::sparcv9.
class LLVM_LIBRARY_VISIBILITY SparcV9TargetInfo : public SparcTargetInfo {
public:
  SparcV9TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : SparcTargetInfo(Triple, Opts) {
    // FIXME: Support Sparc quad-precision long double?
    resetDataLayout();
    // This is an LP64 platform.
    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;

    // OpenBSD uses long long for int64_t and intmax_t.
    if (getTriple().isOSOpenBSD())
      IntMaxType = SignedLongLong;
    else
      IntMaxType = SignedLong;
    Int64Type = IntMaxType;

    // The SPARCv8 System V ABI has long double 128-bits in size, but 64-bit
    // aligned. The SPARCv9 SCD 2.4.1 says 16-byte aligned.
    LongDoubleWidth = 128;
    LongDoubleAlign = 128;
    SuitableAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool isValidCPUName(StringRef Name) const override;

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;

  bool hasBitIntType() const override { return true; }
};
} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_SPARC_H
