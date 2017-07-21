//===--- ARM.h - Declare ARM target feature support -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares ARM TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_ARM_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_ARM_H

#include "OSTargets.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TargetParser.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY ARMTargetInfo : public TargetInfo {
  // Possible FPU choices.
  enum FPUMode {
    VFP2FPU = (1 << 0),
    VFP3FPU = (1 << 1),
    VFP4FPU = (1 << 2),
    NeonFPU = (1 << 3),
    FPARMV8 = (1 << 4)
  };

  // Possible HWDiv features.
  enum HWDivMode { HWDivThumb = (1 << 0), HWDivARM = (1 << 1) };

  static bool FPUModeIsVFP(FPUMode Mode) {
    return Mode & (VFP2FPU | VFP3FPU | VFP4FPU | NeonFPU | FPARMV8);
  }

  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char *const GCCRegNames[];

  std::string ABI, CPU;

  StringRef CPUProfile;
  StringRef CPUAttr;

  enum { FP_Default, FP_VFP, FP_Neon } FPMath;

  unsigned ArchISA;
  unsigned ArchKind = llvm::ARM::AK_ARMV4T;
  unsigned ArchProfile;
  unsigned ArchVersion;

  unsigned FPU : 5;

  unsigned IsAAPCS : 1;
  unsigned HWDiv : 2;

  // Initialized via features.
  unsigned SoftFloat : 1;
  unsigned SoftFloatABI : 1;

  unsigned CRC : 1;
  unsigned Crypto : 1;
  unsigned DSP : 1;
  unsigned Unaligned : 1;

  enum {
    LDREX_B = (1 << 0), /// byte (8-bit)
    LDREX_H = (1 << 1), /// half (16-bit)
    LDREX_W = (1 << 2), /// word (32-bit)
    LDREX_D = (1 << 3), /// double (64-bit)
  };

  uint32_t LDREX;

  // ACLE 6.5.1 Hardware floating point
  enum {
    HW_FP_HP = (1 << 1), /// half (16-bit)
    HW_FP_SP = (1 << 2), /// single (32-bit)
    HW_FP_DP = (1 << 3), /// double (64-bit)
  };
  uint32_t HW_FP;

  static const Builtin::Info BuiltinInfo[];

  void setABIAAPCS() {
    IsAAPCS = true;

    DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 64;
    const llvm::Triple &T = getTriple();

    // size_t is unsigned long on MachO-derived environments, NetBSD, and
    // OpenBSD.
    if (T.isOSBinFormatMachO() || T.getOS() == llvm::Triple::NetBSD ||
        T.getOS() == llvm::Triple::OpenBSD)
      SizeType = UnsignedLong;
    else
      SizeType = UnsignedInt;

    switch (T.getOS()) {
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
      WCharType = SignedInt;
      break;
    case llvm::Triple::Win32:
      WCharType = UnsignedShort;
      break;
    case llvm::Triple::Linux:
    default:
      // AAPCS 7.1.1, ARM-Linux ABI 2.4: type of wchar_t is unsigned int.
      WCharType = UnsignedInt;
      break;
    }

    UseBitFieldTypeAlignment = true;

    ZeroLengthBitfieldBoundary = 0;

    // Thumb1 add sp, #imm requires the immediate value be multiple of 4,
    // so set preferred for small types to 32.
    if (T.isOSBinFormatMachO()) {
      resetDataLayout(BigEndian
                          ? "E-m:o-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
                          : "e-m:o-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
    } else if (T.isOSWindows()) {
      assert(!BigEndian && "Windows on ARM does not support big endian");
      resetDataLayout("e"
                      "-m:w"
                      "-p:32:32"
                      "-i64:64"
                      "-v128:64:128"
                      "-a:0:32"
                      "-n32"
                      "-S64");
    } else if (T.isOSNaCl()) {
      assert(!BigEndian && "NaCl on ARM does not support big endian");
      resetDataLayout("e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S128");
    } else {
      resetDataLayout(BigEndian
                          ? "E-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
                          : "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
    }

    // FIXME: Enumerated types are variable width in straight AAPCS.
  }

  void setABIAPCS(bool IsAAPCS16) {
    const llvm::Triple &T = getTriple();

    IsAAPCS = false;

    if (IsAAPCS16)
      DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 64;
    else
      DoubleAlign = LongLongAlign = LongDoubleAlign = SuitableAlign = 32;

    // size_t is unsigned int on FreeBSD.
    if (T.getOS() == llvm::Triple::FreeBSD)
      SizeType = UnsignedInt;
    else
      SizeType = UnsignedLong;

    // Revert to using SignedInt on apcs-gnu to comply with existing behaviour.
    WCharType = SignedInt;

    // Do not respect the alignment of bit-field types when laying out
    // structures. This corresponds to PCC_BITFIELD_TYPE_MATTERS in gcc.
    UseBitFieldTypeAlignment = false;

    /// gcc forces the alignment to 4 bytes, regardless of the type of the
    /// zero length bitfield.  This corresponds to EMPTY_FIELD_BOUNDARY in
    /// gcc.
    ZeroLengthBitfieldBoundary = 32;

    if (T.isOSBinFormatMachO() && IsAAPCS16) {
      assert(!BigEndian && "AAPCS16 does not support big-endian");
      resetDataLayout("e-m:o-p:32:32-i64:64-a:0:32-n32-S128");
    } else if (T.isOSBinFormatMachO())
      resetDataLayout(
          BigEndian
              ? "E-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
              : "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32");
    else
      resetDataLayout(
          BigEndian
              ? "E-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
              : "e-m:e-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32");

    // FIXME: Override "preferred align" for double and long long.
  }

  void setArchInfo() {
    StringRef ArchName = getTriple().getArchName();

    ArchISA = llvm::ARM::parseArchISA(ArchName);
    CPU = llvm::ARM::getDefaultCPU(ArchName);
    unsigned AK = llvm::ARM::parseArch(ArchName);
    if (AK != llvm::ARM::AK_INVALID)
      ArchKind = AK;
    setArchInfo(ArchKind);
  }

  void setArchInfo(unsigned Kind) {
    StringRef SubArch;

    // cache TargetParser info
    ArchKind = Kind;
    SubArch = llvm::ARM::getSubArch(ArchKind);
    ArchProfile = llvm::ARM::parseArchProfile(SubArch);
    ArchVersion = llvm::ARM::parseArchVersion(SubArch);

    // cache CPU related strings
    CPUAttr = getCPUAttr();
    CPUProfile = getCPUProfile();
  }

  void setAtomic() {
    // when triple does not specify a sub arch,
    // then we are not using inline atomics
    bool ShouldUseInlineAtomic =
        (ArchISA == llvm::ARM::IK_ARM && ArchVersion >= 6) ||
        (ArchISA == llvm::ARM::IK_THUMB && ArchVersion >= 7);
    // Cortex M does not support 8 byte atomics, while general Thumb2 does.
    if (ArchProfile == llvm::ARM::PK_M) {
      MaxAtomicPromoteWidth = 32;
      if (ShouldUseInlineAtomic)
        MaxAtomicInlineWidth = 32;
    } else {
      MaxAtomicPromoteWidth = 64;
      if (ShouldUseInlineAtomic)
        MaxAtomicInlineWidth = 64;
    }
  }

  bool isThumb() const { return (ArchISA == llvm::ARM::IK_THUMB); }

  bool supportsThumb() const { return CPUAttr.count('T') || ArchVersion >= 6; }

  bool supportsThumb2() const {
    return CPUAttr.equals("6T2") ||
           (ArchVersion >= 7 && !CPUAttr.equals("8M_BASE"));
  }

  StringRef getCPUAttr() const {
    // For most sub-arches, the build attribute CPU name is enough.
    // For Cortex variants, it's slightly different.
    switch (ArchKind) {
    default:
      return llvm::ARM::getCPUAttr(ArchKind);
    case llvm::ARM::AK_ARMV6M:
      return "6M";
    case llvm::ARM::AK_ARMV7S:
      return "7S";
    case llvm::ARM::AK_ARMV7A:
      return "7A";
    case llvm::ARM::AK_ARMV7R:
      return "7R";
    case llvm::ARM::AK_ARMV7M:
      return "7M";
    case llvm::ARM::AK_ARMV7EM:
      return "7EM";
    case llvm::ARM::AK_ARMV7VE:
      return "7VE";
    case llvm::ARM::AK_ARMV8A:
      return "8A";
    case llvm::ARM::AK_ARMV8_1A:
      return "8_1A";
    case llvm::ARM::AK_ARMV8_2A:
      return "8_2A";
    case llvm::ARM::AK_ARMV8MBaseline:
      return "8M_BASE";
    case llvm::ARM::AK_ARMV8MMainline:
      return "8M_MAIN";
    case llvm::ARM::AK_ARMV8R:
      return "8R";
    }
  }

  StringRef getCPUProfile() const {
    switch (ArchProfile) {
    case llvm::ARM::PK_A:
      return "A";
    case llvm::ARM::PK_R:
      return "R";
    case llvm::ARM::PK_M:
      return "M";
    default:
      return "";
    }
  }

public:
  ARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TargetInfo(Triple), FPMath(FP_Default), IsAAPCS(true), LDREX(0),
        HW_FP(0) {

    switch (getTriple().getOS()) {
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
      PtrDiffType = SignedLong;
      break;
    default:
      PtrDiffType = SignedInt;
      break;
    }

    // Cache arch related info.
    setArchInfo();

    // {} in inline assembly are neon specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;

    // FIXME: This duplicates code from the driver that sets the -target-abi
    // option - this code is used if -target-abi isn't passed and should
    // be unified in some way.
    if (Triple.isOSBinFormatMachO()) {
      // The backend is hardwired to assume AAPCS for M-class processors, ensure
      // the frontend matches that.
      if (Triple.getEnvironment() == llvm::Triple::EABI ||
          Triple.getOS() == llvm::Triple::UnknownOS ||
          ArchProfile == llvm::ARM::PK_M) {
        setABI("aapcs");
      } else if (Triple.isWatchABI()) {
        setABI("aapcs16");
      } else {
        setABI("apcs-gnu");
      }
    } else if (Triple.isOSWindows()) {
      // FIXME: this is invalid for WindowsCE
      setABI("aapcs");
    } else {
      // Select the default based on the platform.
      switch (Triple.getEnvironment()) {
      case llvm::Triple::Android:
      case llvm::Triple::GNUEABI:
      case llvm::Triple::GNUEABIHF:
      case llvm::Triple::MuslEABI:
      case llvm::Triple::MuslEABIHF:
        setABI("aapcs-linux");
        break;
      case llvm::Triple::EABIHF:
      case llvm::Triple::EABI:
        setABI("aapcs");
        break;
      case llvm::Triple::GNU:
        setABI("apcs-gnu");
        break;
      default:
        if (Triple.getOS() == llvm::Triple::NetBSD)
          setABI("apcs-gnu");
        else if (Triple.getOS() == llvm::Triple::OpenBSD)
          setABI("aapcs-linux");
        else
          setABI("aapcs");
        break;
      }
    }

    // ARM targets default to using the ARM C++ ABI.
    TheCXXABI.set(TargetCXXABI::GenericARM);

    // ARM has atomics up to 8 bytes
    setAtomic();

    // Maximum alignment for ARM NEON data types should be 64-bits (AAPCS)
    if (IsAAPCS && (Triple.getEnvironment() != llvm::Triple::Android))
      MaxVectorAlign = 64;

    // Do force alignment of members that follow zero length bitfields.  If
    // the alignment of the zero-length bitfield is greater than the member
    // that follows it, `bar', `bar' will be aligned as the  type of the
    // zero length bitfield.
    UseZeroLengthBitfieldAlignment = true;

    if (Triple.getOS() == llvm::Triple::Linux ||
        Triple.getOS() == llvm::Triple::UnknownOS)
      this->MCountName = Opts.EABIVersion == llvm::EABI::GNU
                             ? "\01__gnu_mcount_nc"
                             : "\01mcount";
  }

  StringRef getABI() const override { return ABI; }

  bool setABI(const std::string &Name) override {
    ABI = Name;

    // The defaults (above) are for AAPCS, check if we need to change them.
    //
    // FIXME: We need support for -meabi... we could just mangle it into the
    // name.
    if (Name == "apcs-gnu" || Name == "aapcs16") {
      setABIAPCS(Name == "aapcs16");
      return true;
    }
    if (Name == "aapcs" || Name == "aapcs-vfp" || Name == "aapcs-linux") {
      setABIAAPCS();
      return true;
    }
    return false;
  }

  // FIXME: This should be based on Arch attributes, not CPU names.
  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {

    std::vector<StringRef> TargetFeatures;
    unsigned Arch = llvm::ARM::parseArch(getTriple().getArchName());

    // get default FPU features
    unsigned FPUKind = llvm::ARM::getDefaultFPU(CPU, Arch);
    llvm::ARM::getFPUFeatures(FPUKind, TargetFeatures);

    // get default Extension features
    unsigned Extensions = llvm::ARM::getDefaultExtensions(CPU, Arch);
    llvm::ARM::getExtensionFeatures(Extensions, TargetFeatures);

    for (auto Feature : TargetFeatures)
      if (Feature[0] == '+')
        Features[Feature.drop_front(1)] = true;

    // Enable or disable thumb-mode explicitly per function to enable mixed
    // ARM and Thumb code generation.
    if (isThumb())
      Features["thumb-mode"] = true;
    else
      Features["thumb-mode"] = false;

    // Convert user-provided arm and thumb GNU target attributes to
    // [-|+]thumb-mode target features respectively.
    std::vector<std::string> UpdatedFeaturesVec(FeaturesVec);
    for (auto &Feature : UpdatedFeaturesVec) {
      if (Feature.compare("+arm") == 0)
        Feature = "-thumb-mode";
      else if (Feature.compare("+thumb") == 0)
        Feature = "+thumb-mode";
    }

    return TargetInfo::initFeatureMap(Features, Diags, CPU, UpdatedFeaturesVec);
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;

  bool hasFeature(StringRef Feature) const override;
  bool isValidCPUName(StringRef Name) const override {
    return Name == "generic" ||
           llvm::ARM::parseCPUArch(Name) != llvm::ARM::AK_INVALID;
  }

  bool setCPU(const std::string &Name) override {
    if (Name != "generic")
      setArchInfo(llvm::ARM::parseCPUArch(Name));

    if (ArchKind == llvm::ARM::AK_INVALID)
      return false;
    setAtomic();
    CPU = Name;
    return true;
  }

  bool setFPMath(StringRef Name) override;

  void getTargetDefinesARMV81A(const LangOptions &Opts,
                               MacroBuilder &Builder) const;

  void getTargetDefinesARMV82A(const LangOptions &Opts,
                               MacroBuilder &Builder) const {
    // Also include the ARMv8.1-A defines
    getTargetDefinesARMV81A(Opts, Builder);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
  ArrayRef<Builtin::Info> getTargetBuiltins() const override;
  bool isCLZForZeroUndef() const override { return false; }
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return IsAAPCS
               ? AAPCSABIBuiltinVaList
               : (getTriple().isWatchABI() ? TargetInfo::CharPtrBuiltinVaList
                                           : TargetInfo::VoidPtrBuiltinVaList);
  }
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      break;
    case 'l': // r0-r7
    case 'h': // r8-r15
    case 't': // VFP Floating point register single precision
    case 'w': // VFP Floating point register double precision
      Info.setAllowsRegister();
      return true;
    case 'I':
    case 'J':
    case 'K':
    case 'L':
    case 'M':
      // FIXME
      return true;
    case 'Q': // A memory address that is a single base register.
      Info.setAllowsMemory();
      return true;
    case 'U': // a memory reference...
      switch (Name[1]) {
      case 'q': // ...ARMV4 ldrsb
      case 'v': // ...VFP load/store (reg+constant offset)
      case 'y': // ...iWMMXt load/store
      case 't': // address valid for load/store opaque types wider
                // than 128-bits
      case 'n': // valid address for Neon doubleword vector load/store
      case 'm': // valid address for Neon element and structure load/store
      case 's': // valid address for non-offset loads/stores of quad-word
                // values in four ARM registers
        Info.setAllowsMemory();
        Name++;
        return true;
      }
    }
    return false;
  }
  std::string convertConstraint(const char *&Constraint) const override {
    std::string R;
    switch (*Constraint) {
    case 'U': // Two-character constraint; add "^" hint for later parsing.
      R = std::string("^") + std::string(Constraint, 2);
      Constraint++;
      break;
    case 'p': // 'p' should be translated to 'r' by default.
      R = std::string("r");
      break;
    default:
      return std::string(1, *Constraint);
    }
    return R;
  }
  bool
  validateConstraintModifier(StringRef Constraint, char Modifier, unsigned Size,
                             std::string &SuggestedModifier) const override {
    bool isOutput = (Constraint[0] == '=');
    bool isInOut = (Constraint[0] == '+');

    // Strip off constraint modifiers.
    while (Constraint[0] == '=' || Constraint[0] == '+' || Constraint[0] == '&')
      Constraint = Constraint.substr(1);

    switch (Constraint[0]) {
    default:
      break;
    case 'r': {
      switch (Modifier) {
      default:
        return (isInOut || isOutput || Size <= 64);
      case 'q':
        // A register of size 32 cannot fit a vector type.
        return false;
      }
    }
    }

    return true;
  }
  const char *getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_AAPCS:
    case CC_AAPCS_VFP:
    case CC_Swift:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0)
      return 0;
    if (RegNo == 1)
      return 1;
    return -1;
  }

  bool hasSjLjLowering() const override { return true; }
};

class LLVM_LIBRARY_VISIBILITY ARMleTargetInfo : public ARMTargetInfo {
public:
  ARMleTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : ARMTargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

class LLVM_LIBRARY_VISIBILITY ARMbeTargetInfo : public ARMTargetInfo {
public:
  ARMbeTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : ARMTargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

class LLVM_LIBRARY_VISIBILITY WindowsARMTargetInfo
    : public WindowsTargetInfo<ARMleTargetInfo> {
  const llvm::Triple Triple;

public:
  WindowsARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsTargetInfo<ARMleTargetInfo>(Triple, Opts), Triple(Triple) {
    WCharType = UnsignedShort;
    SizeType = UnsignedInt;
  }
  void getVisualStudioDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const;
  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_X86StdCall:
    case CC_X86ThisCall:
    case CC_X86FastCall:
    case CC_X86VectorCall:
      return CCCR_Ignore;
    case CC_C:
    case CC_OpenCLKernel:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }
};

// Windows ARM + Itanium C++ ABI Target
class LLVM_LIBRARY_VISIBILITY ItaniumWindowsARMleTargetInfo
    : public WindowsARMTargetInfo {
public:
  ItaniumWindowsARMleTargetInfo(const llvm::Triple &Triple,
                                const TargetOptions &Opts)
      : WindowsARMTargetInfo(Triple, Opts) {
    TheCXXABI.set(TargetCXXABI::GenericARM);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsARMTargetInfo::getTargetDefines(Opts, Builder);

    if (Opts.MSVCCompat)
      WindowsARMTargetInfo::getVisualStudioDefines(Opts, Builder);
  }
};

// Windows ARM, MS (C++) ABI
class LLVM_LIBRARY_VISIBILITY MicrosoftARMleTargetInfo
    : public WindowsARMTargetInfo {
public:
  MicrosoftARMleTargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : WindowsARMTargetInfo(Triple, Opts) {
    TheCXXABI.set(TargetCXXABI::Microsoft);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsARMTargetInfo::getTargetDefines(Opts, Builder);
    WindowsARMTargetInfo::getVisualStudioDefines(Opts, Builder);
  }
};

// ARM MinGW target
class LLVM_LIBRARY_VISIBILITY MinGWARMTargetInfo : public WindowsARMTargetInfo {
public:
  MinGWARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : WindowsARMTargetInfo(Triple, Opts) {
    TheCXXABI.set(TargetCXXABI::GenericARM);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

// ARM Cygwin target
class LLVM_LIBRARY_VISIBILITY CygwinARMTargetInfo : public ARMleTargetInfo {
public:
  CygwinARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : ARMleTargetInfo(Triple, Opts) {
    TLSSupported = false;
    WCharType = UnsignedShort;
    DoubleAlign = LongLongAlign = 64;
    resetDataLayout("e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

class LLVM_LIBRARY_VISIBILITY DarwinARMTargetInfo
    : public DarwinTargetInfo<ARMleTargetInfo> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    getDarwinDefines(Builder, Opts, Triple, PlatformName, PlatformMinVersion);
  }

public:
  DarwinARMTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<ARMleTargetInfo>(Triple, Opts) {
    HasAlignMac68kSupport = true;
    // iOS always has 64-bit atomic instructions.
    // FIXME: This should be based off of the target features in
    // ARMleTargetInfo.
    MaxAtomicInlineWidth = 64;

    if (Triple.isWatchABI()) {
      // Darwin on iOS uses a variant of the ARM C++ ABI.
      TheCXXABI.set(TargetCXXABI::WatchOS);

      // The 32-bit ABI is silent on what ptrdiff_t should be, but given that
      // size_t is long, it's a bit weird for it to be int.
      PtrDiffType = SignedLong;

      // BOOL should be a real boolean on the new ABI
      UseSignedCharForObjCBool = false;
    } else
      TheCXXABI.set(TargetCXXABI::iOS);
  }
};
// 32-bit RenderScript is armv7 with width and align of 'long' set to 8-bytes
class LLVM_LIBRARY_VISIBILITY RenderScript32TargetInfo
    : public ARMleTargetInfo {
public:
  RenderScript32TargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : ARMleTargetInfo(llvm::Triple("armv7", Triple.getVendorName(),
                                     Triple.getOSName(),
                                     Triple.getEnvironmentName()),
                        Opts) {
    IsRenderScriptTarget = true;
    LongWidth = LongAlign = 64;
  }
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_ARM_H
