//===--- AArch64.h - Declare AArch64 target feature support -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares AArch64 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_AARCH64_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_AARCH64_H

#include "OSTargets.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Support/TargetParser.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY AArch64TargetInfo : public TargetInfo {
  virtual void setDataLayout() = 0;
  static const TargetInfo::GCCRegAlias GCCRegAliases[];
  static const char *const GCCRegNames[];

  enum FPUModeEnum { FPUMode, NeonMode = (1 << 0), SveMode = (1 << 1) };

  unsigned FPU;
  unsigned CRC;
  unsigned Crypto;
  unsigned Unaligned;
  unsigned HasFullFP16;
  llvm::AArch64::ArchKind ArchKind;

  static const Builtin::Info BuiltinInfo[];

  std::string ABI;

public:
  AArch64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TargetInfo(Triple), ABI("aapcs") {
    if (getTriple().getOS() == llvm::Triple::NetBSD ||
        getTriple().getOS() == llvm::Triple::OpenBSD) {
      WCharType = SignedInt;

      // NetBSD apparently prefers consistency across ARM targets to consistency
      // across 64-bit targets.
      Int64Type = SignedLongLong;
      IntMaxType = SignedLongLong;
    } else {
      WCharType = UnsignedInt;
      Int64Type = SignedLong;
      IntMaxType = SignedLong;
    }

    LongWidth = LongAlign = PointerWidth = PointerAlign = 64;
    MaxVectorAlign = 128;
    MaxAtomicInlineWidth = 128;
    MaxAtomicPromoteWidth = 128;

    LongDoubleWidth = LongDoubleAlign = SuitableAlign = 128;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();

    // Make __builtin_ms_va_list available.
    HasBuiltinMSVaList = true;

    // {} in inline assembly are neon specifiers, not assembly variant
    // specifiers.
    NoAsmVariants = true;

    // AAPCS gives rules for bitfields. 7.1.7 says: "The container type
    // contributes to the alignment of the containing aggregate in the same way
    // a plain (non bit-field) member of that type would, without exception for
    // zero-sized or anonymous bit-fields."
    assert(UseBitFieldTypeAlignment && "bitfields affect type alignment");
    UseZeroLengthBitfieldAlignment = true;

    // AArch64 targets default to using the ARM C++ ABI.
    TheCXXABI.set(TargetCXXABI::GenericAArch64);

    if (Triple.getOS() == llvm::Triple::Linux)
      this->MCountName = "\01_mcount";
    else if (Triple.getOS() == llvm::Triple::UnknownOS)
      this->MCountName =
          Opts.EABIVersion == llvm::EABI::GNU ? "\01_mcount" : "mcount";
  }

  StringRef getABI() const override { return ABI; }
  bool setABI(const std::string &Name) override {
    if (Name != "aapcs" && Name != "darwinpcs")
      return false;

    ABI = Name;
    return true;
  }

  bool isValidCPUName(StringRef Name) const override {
    return Name == "generic" ||
           llvm::AArch64::parseCPUArch(Name) !=
               static_cast<unsigned>(llvm::AArch64::ArchKind::AK_INVALID);
  }

  bool setCPU(const std::string &Name) override { return isValidCPUName(Name); }

  void getTargetDefinesARMV81A(const LangOptions &Opts,
                               MacroBuilder &Builder) const {
    Builder.defineMacro("__ARM_FEATURE_QRDMX", "1");
  }

  void getTargetDefinesARMV82A(const LangOptions &Opts,
                               MacroBuilder &Builder) const {
    // Also include the ARMv8.1 defines
    getTargetDefinesARMV81A(Opts, Builder);
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    // Target identification.
    Builder.defineMacro("__aarch64__");
    // For bare-metal none-eabi.
    if (getTriple().getOS() == llvm::Triple::UnknownOS &&
        (getTriple().getEnvironment() == llvm::Triple::EABI ||
         getTriple().getEnvironment() == llvm::Triple::EABIHF))
      Builder.defineMacro("__ELF__");

    // Target properties.
    Builder.defineMacro("_LP64");
    Builder.defineMacro("__LP64__");

    // ACLE predefines. Many can only have one possible value on v8 AArch64.
    Builder.defineMacro("__ARM_ACLE", "200");
    Builder.defineMacro("__ARM_ARCH", "8");
    Builder.defineMacro("__ARM_ARCH_PROFILE", "'A'");

    Builder.defineMacro("__ARM_64BIT_STATE", "1");
    Builder.defineMacro("__ARM_PCS_AAPCS64", "1");
    Builder.defineMacro("__ARM_ARCH_ISA_A64", "1");

    Builder.defineMacro("__ARM_FEATURE_CLZ", "1");
    Builder.defineMacro("__ARM_FEATURE_FMA", "1");
    Builder.defineMacro("__ARM_FEATURE_LDREX", "0xF");
    Builder.defineMacro("__ARM_FEATURE_IDIV", "1"); // As specified in ACLE
    Builder.defineMacro("__ARM_FEATURE_DIV"); // For backwards compatibility
    Builder.defineMacro("__ARM_FEATURE_NUMERIC_MAXMIN", "1");
    Builder.defineMacro("__ARM_FEATURE_DIRECTED_ROUNDING", "1");

    Builder.defineMacro("__ARM_ALIGN_MAX_STACK_PWR", "4");

    // 0xe implies support for half, single and double precision operations.
    Builder.defineMacro("__ARM_FP", "0xE");

    // PCS specifies this for SysV variants, which is all we support. Other ABIs
    // may choose __ARM_FP16_FORMAT_ALTERNATIVE.
    Builder.defineMacro("__ARM_FP16_FORMAT_IEEE", "1");
    Builder.defineMacro("__ARM_FP16_ARGS", "1");

    if (Opts.UnsafeFPMath)
      Builder.defineMacro("__ARM_FP_FAST", "1");

    Builder.defineMacro("__ARM_SIZEOF_WCHAR_T", Opts.ShortWChar ? "2" : "4");

    Builder.defineMacro("__ARM_SIZEOF_MINIMAL_ENUM",
                        Opts.ShortEnums ? "1" : "4");

    if (FPU & NeonMode) {
      Builder.defineMacro("__ARM_NEON", "1");
      // 64-bit NEON supports half, single and double precision operations.
      Builder.defineMacro("__ARM_NEON_FP", "0xE");
    }

    if (FPU & SveMode)
      Builder.defineMacro("__ARM_FEATURE_SVE", "1");

    if (CRC)
      Builder.defineMacro("__ARM_FEATURE_CRC32", "1");

    if (Crypto)
      Builder.defineMacro("__ARM_FEATURE_CRYPTO", "1");

    if (Unaligned)
      Builder.defineMacro("__ARM_FEATURE_UNALIGNED", "1");

    switch (ArchKind) {
    default:
      break;
    case llvm::AArch64::ArchKind::AK_ARMV8_1A:
      getTargetDefinesARMV81A(Opts, Builder);
      break;
    case llvm::AArch64::ArchKind::AK_ARMV8_2A:
      getTargetDefinesARMV82A(Opts, Builder);
      break;
    }

    // All of the __sync_(bool|val)_compare_and_swap_(1|2|4|8) builtins work.
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return llvm::makeArrayRef(BuiltinInfo, clang::AArch64::LastTSBuiltin -
                                               Builtin::FirstTSBuiltin);
  }

  bool hasFeature(StringRef Feature) const override {
    return Feature == "aarch64" || Feature == "arm64" || Feature == "arm" ||
           (Feature == "neon" && (FPU & NeonMode)) ||
           (Feature == "sve" && (FPU & SveMode));
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    FPU = FPUMode;
    CRC = 0;
    Crypto = 0;
    Unaligned = 1;
    HasFullFP16 = 0;
    ArchKind = llvm::AArch64::ArchKind::AK_ARMV8A;

    for (const auto &Feature : Features) {
      if (Feature == "+neon")
        FPU |= NeonMode;
      if (Feature == "+sve")
        FPU |= SveMode;
      if (Feature == "+crc")
        CRC = 1;
      if (Feature == "+crypto")
        Crypto = 1;
      if (Feature == "+strict-align")
        Unaligned = 0;
      if (Feature == "+v8.1a")
        ArchKind = llvm::AArch64::ArchKind::AK_ARMV8_1A;
      if (Feature == "+v8.2a")
        ArchKind = llvm::AArch64::ArchKind::AK_ARMV8_2A;
      if (Feature == "+fullfp16")
        HasFullFP16 = 1;
    }

    setDataLayout();

    return true;
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_C:
    case CC_Swift:
    case CC_PreserveMost:
    case CC_PreserveAll:
    case CC_OpenCLKernel:
    case CC_Win64:
      return CCCR_OK;
    default:
      return CCCR_Warning;
    }
  }

  bool isCLZForZeroUndef() const override { return false; }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::AArch64ABIBuiltinVaList;
  }

  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      return false;
    case 'w': // Floating point and SIMD registers (V0-V31)
      Info.setAllowsRegister();
      return true;
    case 'I': // Constant that can be used with an ADD instruction
    case 'J': // Constant that can be used with a SUB instruction
    case 'K': // Constant that can be used with a 32-bit logical instruction
    case 'L': // Constant that can be used with a 64-bit logical instruction
    case 'M': // Constant that can be used as a 32-bit MOV immediate
    case 'N': // Constant that can be used as a 64-bit MOV immediate
    case 'Y': // Floating point constant zero
    case 'Z': // Integer constant zero
      return true;
    case 'Q': // A memory reference with base register and no offset
      Info.setAllowsMemory();
      return true;
    case 'S': // A symbolic address
      Info.setAllowsRegister();
      return true;
    case 'U':
      // Ump: A memory address suitable for ldp/stp in SI, DI, SF and DF modes.
      // Utf: A memory address suitable for ldp/stp in TF mode.
      // Usa: An absolute symbolic address.
      // Ush: The high part (bits 32:12) of a pc-relative symbolic address.
      llvm_unreachable("FIXME: Unimplemented support for U* constraints.");
    case 'z': // Zero register, wzr or xzr
      Info.setAllowsRegister();
      return true;
    case 'x': // Floating point and SIMD registers (V0-V15)
      Info.setAllowsRegister();
      return true;
    }
    return false;
  }

  bool
  validateConstraintModifier(StringRef Constraint, char Modifier, unsigned Size,
                             std::string &SuggestedModifier) const override {
    // Strip off constraint modifiers.
    while (Constraint[0] == '=' || Constraint[0] == '+' || Constraint[0] == '&')
      Constraint = Constraint.substr(1);

    switch (Constraint[0]) {
    default:
      return true;
    case 'z':
    case 'r': {
      switch (Modifier) {
      case 'x':
      case 'w':
        // For now assume that the person knows what they're
        // doing with the modifier.
        return true;
      default:
        // By default an 'r' constraint will be in the 'x'
        // registers.
        if (Size == 64)
          return true;

        SuggestedModifier = "w";
        return false;
      }
    }
    }
  }

  const char *getClobbers() const override { return ""; }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    if (RegNo == 0)
      return 0;
    if (RegNo == 1)
      return 1;
    return -1;
  }
};

class LLVM_LIBRARY_VISIBILITY AArch64leTargetInfo : public AArch64TargetInfo {
  void setDataLayout() override {
    if (getTriple().isOSBinFormatMachO())
      resetDataLayout("e-m:o-i64:64-i128:128-n32:64-S128");
    else
      resetDataLayout("e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  }

public:
  AArch64leTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : AArch64TargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__AARCH64EL__");
    AArch64TargetInfo::getTargetDefines(Opts, Builder);
  }
};

class LLVM_LIBRARY_VISIBILITY MicrosoftARM64TargetInfo
    : public WindowsTargetInfo<AArch64leTargetInfo> {
  const llvm::Triple Triple;

public:
  MicrosoftARM64TargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : WindowsTargetInfo<AArch64leTargetInfo>(Triple, Opts), Triple(Triple) {

    // This is an LLP64 platform.
    // int:4, long:4, long long:8, long double:8.
    WCharType = UnsignedShort;
    IntWidth = IntAlign = 32;
    LongWidth = LongAlign = 32;
    DoubleAlign = LongLongAlign = 64;
    LongDoubleWidth = LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    IntMaxType = SignedLongLong;
    Int64Type = SignedLongLong;
    SizeType = UnsignedLongLong;
    PtrDiffType = SignedLongLong;
    IntPtrType = SignedLongLong;

    TheCXXABI.set(TargetCXXABI::Microsoft);
  }

  void setDataLayout() override {
    resetDataLayout("e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128");
  }

  void getVisualStudioDefines(const LangOptions &Opts,
                              MacroBuilder &Builder) const {
    WindowsTargetInfo<AArch64leTargetInfo>::getVisualStudioDefines(Opts,
                                                                   Builder);
    Builder.defineMacro("_WIN32", "1");
    Builder.defineMacro("_WIN64", "1");
    Builder.defineMacro("_M_ARM64", "1");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    WindowsTargetInfo::getTargetDefines(Opts, Builder);
    getVisualStudioDefines(Opts, Builder);
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

class LLVM_LIBRARY_VISIBILITY AArch64beTargetInfo : public AArch64TargetInfo {
  void setDataLayout() override {
    assert(!getTriple().isOSBinFormatMachO());
    resetDataLayout("E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128");
  }

public:
  AArch64beTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : AArch64TargetInfo(Triple, Opts) {}
  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__AARCH64EB__");
    Builder.defineMacro("__AARCH_BIG_ENDIAN");
    Builder.defineMacro("__ARM_BIG_ENDIAN");
    AArch64TargetInfo::getTargetDefines(Opts, Builder);
  }
};

class LLVM_LIBRARY_VISIBILITY DarwinAArch64TargetInfo
    : public DarwinTargetInfo<AArch64leTargetInfo> {
protected:
  void getOSDefines(const LangOptions &Opts, const llvm::Triple &Triple,
                    MacroBuilder &Builder) const override {
    Builder.defineMacro("__AARCH64_SIMD__");
    Builder.defineMacro("__ARM64_ARCH_8__");
    Builder.defineMacro("__ARM_NEON__");
    Builder.defineMacro("__LITTLE_ENDIAN__");
    Builder.defineMacro("__REGISTER_PREFIX__", "");
    Builder.defineMacro("__arm64", "1");
    Builder.defineMacro("__arm64__", "1");

    getDarwinDefines(Builder, Opts, Triple, PlatformName, PlatformMinVersion);
  }

public:
  DarwinAArch64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : DarwinTargetInfo<AArch64leTargetInfo>(Triple, Opts) {
    Int64Type = SignedLongLong;
    WCharType = SignedInt;
    UseSignedCharForObjCBool = false;

    LongDoubleWidth = LongDoubleAlign = SuitableAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();

    TheCXXABI.set(TargetCXXABI::iOS64);
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }
};

// 64-bit RenderScript is aarch64
class LLVM_LIBRARY_VISIBILITY RenderScript64TargetInfo
    : public AArch64leTargetInfo {
public:
  RenderScript64TargetInfo(const llvm::Triple &Triple,
                           const TargetOptions &Opts)
      : AArch64leTargetInfo(llvm::Triple("aarch64", Triple.getVendorName(),
                                         Triple.getOSName(),
                                         Triple.getEnvironmentName()),
                            Opts) {
    IsRenderScriptTarget = true;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    Builder.defineMacro("__RENDERSCRIPT__");
    AArch64leTargetInfo::getTargetDefines(Opts, Builder);
  }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_AARCH64_H
