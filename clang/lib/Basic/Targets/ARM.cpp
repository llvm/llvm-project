//===--- ARM.cpp - Implement ARM target feature support -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ARM TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

bool ARMTargetInfo::setFPMath(StringRef Name) {
  if (Name == "neon") {
    FPMath = FP_Neon;
    return true;
  } else if (Name == "vfp" || Name == "vfp2" || Name == "vfp3" ||
             Name == "vfp4") {
    FPMath = FP_VFP;
    return true;
  }
  return false;
}

const char *const ARMTargetInfo::GCCRegNames[] = {
    // Integer registers
    "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11",
    "r12", "sp", "lr", "pc",

    // Float registers
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22",
    "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",

    // Double registers
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
    "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22",
    "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31",

    // Quad registers
    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
    "q12", "q13", "q14", "q15"
};

ArrayRef<const char *> ARMTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias ARMTargetInfo::GCCRegAliases[] = {
    {{"a1"}, "r0"},  {{"a2"}, "r1"},        {{"a3"}, "r2"},  {{"a4"}, "r3"},
    {{"v1"}, "r4"},  {{"v2"}, "r5"},        {{"v3"}, "r6"},  {{"v4"}, "r7"},
    {{"v5"}, "r8"},  {{"v6", "rfp"}, "r9"}, {{"sl"}, "r10"}, {{"fp"}, "r11"},
    {{"ip"}, "r12"}, {{"r13"}, "sp"},       {{"r14"}, "lr"}, {{"r15"}, "pc"},
    // The S, D and Q registers overlap, but aren't really aliases; we
    // don't want to substitute one of these for a different-sized one.
};

ArrayRef<TargetInfo::GCCRegAlias> ARMTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

const Builtin::Info ARMTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsNEON.def"

#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define LANGBUILTIN(ID, TYPE, ATTRS, LANG)                                     \
  {#ID, TYPE, ATTRS, nullptr, LANG, nullptr},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr},
#define TARGET_HEADER_BUILTIN(ID, TYPE, ATTRS, HEADER, LANGS, FEATURE)         \
  {#ID, TYPE, ATTRS, HEADER, LANGS, FEATURE},
#include "clang/Basic/BuiltinsARM.def"
};

bool ARMTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("arm", true)
      .Case("aarch32", true)
      .Case("softfloat", SoftFloat)
      .Case("thumb", isThumb())
      .Case("neon", (FPU & NeonFPU) && !SoftFloat)
      .Case("vfp", FPU && !SoftFloat)
      .Case("hwdiv", HWDiv & HWDivThumb)
      .Case("hwdiv-arm", HWDiv & HWDivARM)
      .Default(false);
}

void ARMTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  Builder.defineMacro("__arm");
  Builder.defineMacro("__arm__");
  // For bare-metal none-eabi.
  if (getTriple().getOS() == llvm::Triple::UnknownOS &&
      (getTriple().getEnvironment() == llvm::Triple::EABI ||
       getTriple().getEnvironment() == llvm::Triple::EABIHF))
    Builder.defineMacro("__ELF__");

  // Target properties.
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // Unfortunately, __ARM_ARCH_7K__ is now more of an ABI descriptor. The CPU
  // happens to be Cortex-A7 though, so it should still get __ARM_ARCH_7A__.
  if (getTriple().isWatchABI())
    Builder.defineMacro("__ARM_ARCH_7K__", "2");

  if (!CPUAttr.empty())
    Builder.defineMacro("__ARM_ARCH_" + CPUAttr + "__");

  // ACLE 6.4.1 ARM/Thumb instruction set architecture
  // __ARM_ARCH is defined as an integer value indicating the current ARM ISA
  Builder.defineMacro("__ARM_ARCH", Twine(ArchVersion));

  if (ArchVersion >= 8) {
    // ACLE 6.5.7 Crypto Extension
    if (Crypto)
      Builder.defineMacro("__ARM_FEATURE_CRYPTO", "1");
    // ACLE 6.5.8 CRC32 Extension
    if (CRC)
      Builder.defineMacro("__ARM_FEATURE_CRC32", "1");
    // ACLE 6.5.10 Numeric Maximum and Minimum
    Builder.defineMacro("__ARM_FEATURE_NUMERIC_MAXMIN", "1");
    // ACLE 6.5.9 Directed Rounding
    Builder.defineMacro("__ARM_FEATURE_DIRECTED_ROUNDING", "1");
  }

  // __ARM_ARCH_ISA_ARM is defined to 1 if the core supports the ARM ISA.  It
  // is not defined for the M-profile.
  // NOTE that the default profile is assumed to be 'A'
  if (CPUProfile.empty() || ArchProfile != llvm::ARM::PK_M)
    Builder.defineMacro("__ARM_ARCH_ISA_ARM", "1");

  // __ARM_ARCH_ISA_THUMB is defined to 1 if the core supports the original
  // Thumb ISA (including v6-M and v8-M Baseline).  It is set to 2 if the
  // core supports the Thumb-2 ISA as found in the v6T2 architecture and all
  // v7 and v8 architectures excluding v8-M Baseline.
  if (supportsThumb2())
    Builder.defineMacro("__ARM_ARCH_ISA_THUMB", "2");
  else if (supportsThumb())
    Builder.defineMacro("__ARM_ARCH_ISA_THUMB", "1");

  // __ARM_32BIT_STATE is defined to 1 if code is being generated for a 32-bit
  // instruction set such as ARM or Thumb.
  Builder.defineMacro("__ARM_32BIT_STATE", "1");

  // ACLE 6.4.2 Architectural Profile (A, R, M or pre-Cortex)

  // __ARM_ARCH_PROFILE is defined as 'A', 'R', 'M' or 'S', or unset.
  if (!CPUProfile.empty())
    Builder.defineMacro("__ARM_ARCH_PROFILE", "'" + CPUProfile + "'");

  // ACLE 6.4.3 Unaligned access supported in hardware
  if (Unaligned)
    Builder.defineMacro("__ARM_FEATURE_UNALIGNED", "1");

  // ACLE 6.4.4 LDREX/STREX
  if (LDREX)
    Builder.defineMacro("__ARM_FEATURE_LDREX", "0x" + llvm::utohexstr(LDREX));

  // ACLE 6.4.5 CLZ
  if (ArchVersion == 5 || (ArchVersion == 6 && CPUProfile != "M") ||
      ArchVersion > 6)
    Builder.defineMacro("__ARM_FEATURE_CLZ", "1");

  // ACLE 6.5.1 Hardware Floating Point
  if (HW_FP)
    Builder.defineMacro("__ARM_FP", "0x" + llvm::utohexstr(HW_FP));

  // ACLE predefines.
  Builder.defineMacro("__ARM_ACLE", "200");

  // FP16 support (we currently only support IEEE format).
  Builder.defineMacro("__ARM_FP16_FORMAT_IEEE", "1");
  Builder.defineMacro("__ARM_FP16_ARGS", "1");

  // ACLE 6.5.3 Fused multiply-accumulate (FMA)
  if (ArchVersion >= 7 && (FPU & VFP4FPU))
    Builder.defineMacro("__ARM_FEATURE_FMA", "1");

  // Subtarget options.

  // FIXME: It's more complicated than this and we don't really support
  // interworking.
  // Windows on ARM does not "support" interworking
  if (5 <= ArchVersion && ArchVersion <= 8 && !getTriple().isOSWindows())
    Builder.defineMacro("__THUMB_INTERWORK__");

  if (ABI == "aapcs" || ABI == "aapcs-linux" || ABI == "aapcs-vfp") {
    // Embedded targets on Darwin follow AAPCS, but not EABI.
    // Windows on ARM follows AAPCS VFP, but does not conform to EABI.
    if (!getTriple().isOSBinFormatMachO() && !getTriple().isOSWindows())
      Builder.defineMacro("__ARM_EABI__");
    Builder.defineMacro("__ARM_PCS", "1");
  }

  if ((!SoftFloat && !SoftFloatABI) || ABI == "aapcs-vfp" || ABI == "aapcs16")
    Builder.defineMacro("__ARM_PCS_VFP", "1");

  if (SoftFloat)
    Builder.defineMacro("__SOFTFP__");

  if (ArchKind == llvm::ARM::AK_XSCALE)
    Builder.defineMacro("__XSCALE__");

  if (isThumb()) {
    Builder.defineMacro("__THUMBEL__");
    Builder.defineMacro("__thumb__");
    if (supportsThumb2())
      Builder.defineMacro("__thumb2__");
  }

  // ACLE 6.4.9 32-bit SIMD instructions
  if (ArchVersion >= 6 && (CPUProfile != "M" || CPUAttr == "7EM"))
    Builder.defineMacro("__ARM_FEATURE_SIMD32", "1");

  // ACLE 6.4.10 Hardware Integer Divide
  if (((HWDiv & HWDivThumb) && isThumb()) ||
      ((HWDiv & HWDivARM) && !isThumb())) {
    Builder.defineMacro("__ARM_FEATURE_IDIV", "1");
    Builder.defineMacro("__ARM_ARCH_EXT_IDIV__", "1");
  }

  // Note, this is always on in gcc, even though it doesn't make sense.
  Builder.defineMacro("__APCS_32__");

  if (FPUModeIsVFP((FPUMode)FPU)) {
    Builder.defineMacro("__VFP_FP__");
    if (FPU & VFP2FPU)
      Builder.defineMacro("__ARM_VFPV2__");
    if (FPU & VFP3FPU)
      Builder.defineMacro("__ARM_VFPV3__");
    if (FPU & VFP4FPU)
      Builder.defineMacro("__ARM_VFPV4__");
    if (FPU & FPARMV8)
      Builder.defineMacro("__ARM_FPV5__");
  }

  // This only gets set when Neon instructions are actually available, unlike
  // the VFP define, hence the soft float and arch check. This is subtly
  // different from gcc, we follow the intent which was that it should be set
  // when Neon instructions are actually available.
  if ((FPU & NeonFPU) && !SoftFloat && ArchVersion >= 7) {
    Builder.defineMacro("__ARM_NEON", "1");
    Builder.defineMacro("__ARM_NEON__");
    // current AArch32 NEON implementations do not support double-precision
    // floating-point even when it is present in VFP.
    Builder.defineMacro("__ARM_NEON_FP",
                        "0x" + llvm::utohexstr(HW_FP & ~HW_FP_DP));
  }

  Builder.defineMacro("__ARM_SIZEOF_WCHAR_T", Opts.ShortWChar ? "2" : "4");

  Builder.defineMacro("__ARM_SIZEOF_MINIMAL_ENUM", Opts.ShortEnums ? "1" : "4");

  if (ArchVersion >= 6 && CPUAttr != "6M" && CPUAttr != "8M_BASE") {
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
  }

  // ACLE 6.4.7 DSP instructions
  if (DSP) {
    Builder.defineMacro("__ARM_FEATURE_DSP", "1");
  }

  // ACLE 6.4.8 Saturation instructions
  bool SAT = false;
  if ((ArchVersion == 6 && CPUProfile != "M") || ArchVersion > 6) {
    Builder.defineMacro("__ARM_FEATURE_SAT", "1");
    SAT = true;
  }

  // ACLE 6.4.6 Q (saturation) flag
  if (DSP || SAT)
    Builder.defineMacro("__ARM_FEATURE_QBIT", "1");

  if (Opts.UnsafeFPMath)
    Builder.defineMacro("__ARM_FP_FAST", "1");

  switch (ArchKind) {
  default:
    break;
  case llvm::ARM::AK_ARMV8_1A:
    getTargetDefinesARMV81A(Opts, Builder);
    break;
  case llvm::ARM::AK_ARMV8_2A:
    getTargetDefinesARMV82A(Opts, Builder);
    break;
  }
}

bool ARMTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                         DiagnosticsEngine &Diags) {
  FPU = 0;
  CRC = 0;
  Crypto = 0;
  DSP = 0;
  Unaligned = 1;
  SoftFloat = SoftFloatABI = false;
  HWDiv = 0;

  // This does not diagnose illegal cases like having both
  // "+vfpv2" and "+vfpv3" or having "+neon" and "+fp-only-sp".
  uint32_t HW_FP_remove = 0;
  for (const auto &Feature : Features) {
    if (Feature == "+soft-float") {
      SoftFloat = true;
    } else if (Feature == "+soft-float-abi") {
      SoftFloatABI = true;
    } else if (Feature == "+vfp2") {
      FPU |= VFP2FPU;
      HW_FP |= HW_FP_SP | HW_FP_DP;
    } else if (Feature == "+vfp3") {
      FPU |= VFP3FPU;
      HW_FP |= HW_FP_SP | HW_FP_DP;
    } else if (Feature == "+vfp4") {
      FPU |= VFP4FPU;
      HW_FP |= HW_FP_SP | HW_FP_DP | HW_FP_HP;
    } else if (Feature == "+fp-armv8") {
      FPU |= FPARMV8;
      HW_FP |= HW_FP_SP | HW_FP_DP | HW_FP_HP;
    } else if (Feature == "+neon") {
      FPU |= NeonFPU;
      HW_FP |= HW_FP_SP | HW_FP_DP;
    } else if (Feature == "+hwdiv") {
      HWDiv |= HWDivThumb;
    } else if (Feature == "+hwdiv-arm") {
      HWDiv |= HWDivARM;
    } else if (Feature == "+crc") {
      CRC = 1;
    } else if (Feature == "+crypto") {
      Crypto = 1;
    } else if (Feature == "+dsp") {
      DSP = 1;
    } else if (Feature == "+fp-only-sp") {
      HW_FP_remove |= HW_FP_DP;
    } else if (Feature == "+strict-align") {
      Unaligned = 0;
    } else if (Feature == "+fp16") {
      HW_FP |= HW_FP_HP;
    }
  }
  HW_FP &= ~HW_FP_remove;

  switch (ArchVersion) {
  case 6:
    if (ArchProfile == llvm::ARM::PK_M)
      LDREX = 0;
    else if (ArchKind == llvm::ARM::AK_ARMV6K)
      LDREX = LDREX_D | LDREX_W | LDREX_H | LDREX_B;
    else
      LDREX = LDREX_W;
    break;
  case 7:
    if (ArchProfile == llvm::ARM::PK_M)
      LDREX = LDREX_W | LDREX_H | LDREX_B;
    else
      LDREX = LDREX_D | LDREX_W | LDREX_H | LDREX_B;
    break;
  case 8:
    LDREX = LDREX_D | LDREX_W | LDREX_H | LDREX_B;
  }

  if (!(FPU & NeonFPU) && FPMath == FP_Neon) {
    Diags.Report(diag::err_target_unsupported_fpmath) << "neon";
    return false;
  }

  if (FPMath == FP_Neon)
    Features.push_back("+neonfp");
  else if (FPMath == FP_VFP)
    Features.push_back("-neonfp");

  // Remove front-end specific options which the backend handles differently.
  auto Feature = std::find(Features.begin(), Features.end(), "+soft-float-abi");
  if (Feature != Features.end())
    Features.erase(Feature);

  return true;
}

void ARMTargetInfo::getTargetDefinesARMV81A(const LangOptions &Opts,
                                            MacroBuilder &Builder) const {
  Builder.defineMacro("__ARM_FEATURE_QRDMX", "1");
}

void ARMleTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__ARMEL__");
  ARMTargetInfo::getTargetDefines(Opts, Builder);
}

void ARMbeTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__ARMEB__");
  Builder.defineMacro("__ARM_BIG_ENDIAN");
  ARMTargetInfo::getTargetDefines(Opts, Builder);
}

void WindowsARMTargetInfo::getVisualStudioDefines(const LangOptions &Opts,
                                                  MacroBuilder &Builder) const {
  WindowsTargetInfo<ARMleTargetInfo>::getVisualStudioDefines(Opts, Builder);

  // FIXME: this is invalid for WindowsCE
  Builder.defineMacro("_M_ARM_NT", "1");
  Builder.defineMacro("_M_ARMT", "_M_ARM");
  Builder.defineMacro("_M_THUMB", "_M_ARM");

  assert((Triple.getArch() == llvm::Triple::arm ||
          Triple.getArch() == llvm::Triple::thumb) &&
         "invalid architecture for Windows ARM target info");
  unsigned Offset = Triple.getArch() == llvm::Triple::arm ? 4 : 6;
  Builder.defineMacro("_M_ARM", Triple.getArchName().substr(Offset));

  // TODO map the complete set of values
  // 31: VFPv3 40: VFPv4
  Builder.defineMacro("_M_ARM_FP", "31");
}

void MinGWARMTargetInfo::getTargetDefines(const LangOptions &Opts,
                                          MacroBuilder &Builder) const {
  WindowsARMTargetInfo::getTargetDefines(Opts, Builder);
  DefineStd(Builder, "WIN32", Opts);
  DefineStd(Builder, "WINNT", Opts);
  Builder.defineMacro("_ARM_");
  addMinGWDefines(Opts, Builder);
}

void CygwinARMTargetInfo::getTargetDefines(const LangOptions &Opts,
                                           MacroBuilder &Builder) const {
  ARMleTargetInfo::getTargetDefines(Opts, Builder);
  Builder.defineMacro("_ARM_");
  Builder.defineMacro("__CYGWIN__");
  Builder.defineMacro("__CYGWIN32__");
  DefineStd(Builder, "unix", Opts);
  if (Opts.CPlusPlus)
    Builder.defineMacro("_GNU_SOURCE");
}

void RenderScript32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                                MacroBuilder &Builder) const {
  Builder.defineMacro("__RENDERSCRIPT__");
  ARMleTargetInfo::getTargetDefines(Opts, Builder);
}

ArrayRef<Builtin::Info> ARMTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::ARM::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}
