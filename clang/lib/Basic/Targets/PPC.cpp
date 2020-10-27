//===--- PPC.cpp - Implement PPC target feature support -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements PPC TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "PPC.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::targets;

const Builtin::Info PPCTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, HEADER, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsPPC.def"
};

/// handleTargetFeatures - Perform initialization based on the user
/// configured set of features.
bool PPCTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                         DiagnosticsEngine &Diags) {
  FloatABI = HardFloat;
  for (const auto &Feature : Features) {
    if (Feature == "+altivec") {
      HasAltivec = true;
    } else if (Feature == "+vsx") {
      HasVSX = true;
    } else if (Feature == "+bpermd") {
      HasBPERMD = true;
    } else if (Feature == "+extdiv") {
      HasExtDiv = true;
    } else if (Feature == "+power8-vector") {
      HasP8Vector = true;
    } else if (Feature == "+crypto") {
      HasP8Crypto = true;
    } else if (Feature == "+direct-move") {
      HasDirectMove = true;
    } else if (Feature == "+htm") {
      HasHTM = true;
    } else if (Feature == "+float128") {
      HasFloat128 = true;
    } else if (Feature == "+power9-vector") {
      HasP9Vector = true;
    } else if (Feature == "+power10-vector") {
      HasP10Vector = true;
    } else if (Feature == "+pcrelative-memops") {
      HasPCRelativeMemops = true;
    } else if (Feature == "+spe") {
      HasSPE = true;
      LongDoubleWidth = LongDoubleAlign = 64;
      LongDoubleFormat = &llvm::APFloat::IEEEdouble();
    } else if (Feature == "-hard-float") {
      FloatABI = SoftFloat;
    } else if (Feature == "+paired-vector-memops") {
      PairedVectorMemops = true;
    }
    // TODO: Finish this list and add an assert that we've handled them
    // all.
  }

  return true;
}

/// PPCTargetInfo::getTargetDefines - Return a set of the PowerPC-specific
/// #defines that are not tied to a specific subtarget.
void PPCTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  // Target identification.
  Builder.defineMacro("__ppc__");
  Builder.defineMacro("__PPC__");
  Builder.defineMacro("_ARCH_PPC");
  Builder.defineMacro("__powerpc__");
  Builder.defineMacro("__POWERPC__");
  if (PointerWidth == 64) {
    Builder.defineMacro("_ARCH_PPC64");
    Builder.defineMacro("__powerpc64__");
    Builder.defineMacro("__ppc64__");
    Builder.defineMacro("__PPC64__");
  }

  // Target properties.
  if (getTriple().getArch() == llvm::Triple::ppc64le) {
    Builder.defineMacro("_LITTLE_ENDIAN");
  } else {
    if (!getTriple().isOSNetBSD() &&
        !getTriple().isOSOpenBSD())
      Builder.defineMacro("_BIG_ENDIAN");
  }

  // ABI options.
  if (ABI == "elfv1")
    Builder.defineMacro("_CALL_ELF", "1");
  if (ABI == "elfv2")
    Builder.defineMacro("_CALL_ELF", "2");

  // This typically is only for a new enough linker (bfd >= 2.16.2 or gold), but
  // our support post-dates this and it should work on all 64-bit ppc linux
  // platforms. It is guaranteed to work on all elfv2 platforms.
  if (getTriple().getOS() == llvm::Triple::Linux && PointerWidth == 64)
    Builder.defineMacro("_CALL_LINUX", "1");

  // Subtarget options.
  if (!getTriple().isOSAIX()){
    Builder.defineMacro("__NATURAL_ALIGNMENT__");
  }
  Builder.defineMacro("__REGISTER_PREFIX__", "");

  // FIXME: Should be controlled by command line option.
  if (LongDoubleWidth == 128) {
    Builder.defineMacro("__LONG_DOUBLE_128__");
    Builder.defineMacro("__LONGDOUBLE128");
  }

  // Define this for elfv2 (64-bit only) or 64-bit darwin.
  if (ABI == "elfv2" ||
      (getTriple().getOS() == llvm::Triple::Darwin && PointerWidth == 64))
    Builder.defineMacro("__STRUCT_PARM_ALIGN__", "16");

  if (ArchDefs & ArchDefineName)
    Builder.defineMacro(Twine("_ARCH_", StringRef(CPU).upper()));
  if (ArchDefs & ArchDefinePpcgr)
    Builder.defineMacro("_ARCH_PPCGR");
  if (ArchDefs & ArchDefinePpcsq)
    Builder.defineMacro("_ARCH_PPCSQ");
  if (ArchDefs & ArchDefine440)
    Builder.defineMacro("_ARCH_440");
  if (ArchDefs & ArchDefine603)
    Builder.defineMacro("_ARCH_603");
  if (ArchDefs & ArchDefine604)
    Builder.defineMacro("_ARCH_604");
  if (ArchDefs & ArchDefinePwr4)
    Builder.defineMacro("_ARCH_PWR4");
  if (ArchDefs & ArchDefinePwr5)
    Builder.defineMacro("_ARCH_PWR5");
  if (ArchDefs & ArchDefinePwr5x)
    Builder.defineMacro("_ARCH_PWR5X");
  if (ArchDefs & ArchDefinePwr6)
    Builder.defineMacro("_ARCH_PWR6");
  if (ArchDefs & ArchDefinePwr6x)
    Builder.defineMacro("_ARCH_PWR6X");
  if (ArchDefs & ArchDefinePwr7)
    Builder.defineMacro("_ARCH_PWR7");
  if (ArchDefs & ArchDefinePwr8)
    Builder.defineMacro("_ARCH_PWR8");
  if (ArchDefs & ArchDefinePwr9)
    Builder.defineMacro("_ARCH_PWR9");
  if (ArchDefs & ArchDefinePwr10)
    Builder.defineMacro("_ARCH_PWR10");
  if (ArchDefs & ArchDefineA2)
    Builder.defineMacro("_ARCH_A2");
  if (ArchDefs & ArchDefineE500)
    Builder.defineMacro("__NO_LWSYNC__");
  if (ArchDefs & ArchDefineFuture)
    Builder.defineMacro("_ARCH_PWR_FUTURE");

  if (HasAltivec) {
    Builder.defineMacro("__VEC__", "10206");
    Builder.defineMacro("__ALTIVEC__");
  }
  if (HasSPE) {
    Builder.defineMacro("__SPE__");
    Builder.defineMacro("__NO_FPRS__");
  }
  if (HasVSX)
    Builder.defineMacro("__VSX__");
  if (HasP8Vector)
    Builder.defineMacro("__POWER8_VECTOR__");
  if (HasP8Crypto)
    Builder.defineMacro("__CRYPTO__");
  if (HasHTM)
    Builder.defineMacro("__HTM__");
  if (HasFloat128)
    Builder.defineMacro("__FLOAT128__");
  if (HasP9Vector)
    Builder.defineMacro("__POWER9_VECTOR__");
  if (HasP10Vector)
    Builder.defineMacro("__POWER10_VECTOR__");

  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
  if (PointerWidth == 64)
    Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");

  // We have support for the bswap intrinsics so we can define this.
  Builder.defineMacro("__HAVE_BSWAP__", "1");

  // FIXME: The following are not yet generated here by Clang, but are
  //        generated by GCC:
  //
  //   _SOFT_FLOAT_
  //   __RECIP_PRECISION__
  //   __APPLE_ALTIVEC__
  //   __RECIP__
  //   __RECIPF__
  //   __RSQRTE__
  //   __RSQRTEF__
  //   _SOFT_DOUBLE_
  //   __NO_LWSYNC__
  //   __CMODEL_MEDIUM__
  //   __CMODEL_LARGE__
  //   _CALL_SYSV
  //   _CALL_DARWIN
}

// Handle explicit options being passed to the compiler here: if we've
// explicitly turned off vsx and turned on any of:
// - power8-vector
// - direct-move
// - float128
// - power9-vector
// - paired-vector-memops
// - power10-vector
// then go ahead and error since the customer has expressed an incompatible
// set of options.
static bool ppcUserFeaturesCheck(DiagnosticsEngine &Diags,
                                 const std::vector<std::string> &FeaturesVec) {

  // vsx was not explicitly turned off.
  if (llvm::find(FeaturesVec, "-vsx") == FeaturesVec.end())
    return true;

  auto FindVSXSubfeature = [&](StringRef Feature, StringRef Option) {
    if (llvm::find(FeaturesVec, Feature) != FeaturesVec.end()) {
      Diags.Report(diag::err_opt_not_valid_with_opt) << Option << "-mno-vsx";
      return true;
    }
    return false;
  };

  bool Found = FindVSXSubfeature("+power8-vector", "-mpower8-vector");
  Found |= FindVSXSubfeature("+direct-move", "-mdirect-move");
  Found |= FindVSXSubfeature("+float128", "-mfloat128");
  Found |= FindVSXSubfeature("+power9-vector", "-mpower9-vector");
  Found |= FindVSXSubfeature("+paired-vector-memops", "-mpaired-vector-memops");
  Found |= FindVSXSubfeature("+power10-vector", "-mpower10-vector");

  // Return false if any vsx subfeatures was found.
  return !Found;
}

bool PPCTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {
  Features["altivec"] = llvm::StringSwitch<bool>(CPU)
                            .Case("7400", true)
                            .Case("g4", true)
                            .Case("7450", true)
                            .Case("g4+", true)
                            .Case("970", true)
                            .Case("g5", true)
                            .Case("pwr6", true)
                            .Case("pwr7", true)
                            .Case("pwr8", true)
                            .Case("pwr9", true)
                            .Case("ppc64", true)
                            .Case("ppc64le", true)
                            .Default(false);

  Features["power9-vector"] = (CPU == "pwr9");
  Features["crypto"] = llvm::StringSwitch<bool>(CPU)
                           .Case("ppc64le", true)
                           .Case("pwr9", true)
                           .Case("pwr8", true)
                           .Default(false);
  Features["power8-vector"] = llvm::StringSwitch<bool>(CPU)
                                  .Case("ppc64le", true)
                                  .Case("pwr9", true)
                                  .Case("pwr8", true)
                                  .Default(false);
  Features["bpermd"] = llvm::StringSwitch<bool>(CPU)
                           .Case("ppc64le", true)
                           .Case("pwr9", true)
                           .Case("pwr8", true)
                           .Case("pwr7", true)
                           .Default(false);
  Features["extdiv"] = llvm::StringSwitch<bool>(CPU)
                           .Case("ppc64le", true)
                           .Case("pwr9", true)
                           .Case("pwr8", true)
                           .Case("pwr7", true)
                           .Default(false);
  Features["direct-move"] = llvm::StringSwitch<bool>(CPU)
                                .Case("ppc64le", true)
                                .Case("pwr9", true)
                                .Case("pwr8", true)
                                .Default(false);
  Features["vsx"] = llvm::StringSwitch<bool>(CPU)
                        .Case("ppc64le", true)
                        .Case("pwr9", true)
                        .Case("pwr8", true)
                        .Case("pwr7", true)
                        .Default(false);
  Features["htm"] = llvm::StringSwitch<bool>(CPU)
                        .Case("ppc64le", true)
                        .Case("pwr9", true)
                        .Case("pwr8", true)
                        .Default(false);

  Features["spe"] = llvm::StringSwitch<bool>(CPU)
                        .Case("8548", true)
                        .Case("e500", true)
                        .Default(false);

  // Power10 includes all the same features as Power9 plus any features specific
  // to the Power10 core.
  if (CPU == "pwr10" || CPU == "power10") {
    initFeatureMap(Features, Diags, "pwr9", FeaturesVec);
    addP10SpecificFeatures(Features);
  }

  // Future CPU should include all of the features of Power 10 as well as any
  // additional features (yet to be determined) specific to it.
  if (CPU == "future") {
    initFeatureMap(Features, Diags, "pwr10", FeaturesVec);
    addFutureSpecificFeatures(Features);
  }

  if (!ppcUserFeaturesCheck(Diags, FeaturesVec))
    return false;

  if (!(ArchDefs & ArchDefinePwr9) && (ArchDefs & ArchDefinePpcgr) &&
      llvm::find(FeaturesVec, "+float128") != FeaturesVec.end()) {
    // We have __float128 on PPC but not power 9 and above.
    Diags.Report(diag::err_opt_not_valid_with_opt) << "-mfloat128" << CPU;
    return false;
  }

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

// Add any Power10 specific features.
void PPCTargetInfo::addP10SpecificFeatures(
    llvm::StringMap<bool> &Features) const {
  Features["htm"] = false; // HTM was removed for P10.
  Features["paired-vector-memops"] = true;
  Features["power10-vector"] = true;
  Features["pcrelative-memops"] = true;
  return;
}

// Add features specific to the "Future" CPU.
void PPCTargetInfo::addFutureSpecificFeatures(
    llvm::StringMap<bool> &Features) const {
  return;
}

bool PPCTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("powerpc", true)
      .Case("altivec", HasAltivec)
      .Case("vsx", HasVSX)
      .Case("power8-vector", HasP8Vector)
      .Case("crypto", HasP8Crypto)
      .Case("direct-move", HasDirectMove)
      .Case("htm", HasHTM)
      .Case("bpermd", HasBPERMD)
      .Case("extdiv", HasExtDiv)
      .Case("float128", HasFloat128)
      .Case("power9-vector", HasP9Vector)
      .Case("paired-vector-memops", PairedVectorMemops)
      .Case("power10-vector", HasP10Vector)
      .Case("pcrelative-memops", HasPCRelativeMemops)
      .Case("spe", HasSPE)
      .Default(false);
}

void PPCTargetInfo::setFeatureEnabled(llvm::StringMap<bool> &Features,
                                      StringRef Name, bool Enabled) const {
  if (Enabled) {
    // If we're enabling any of the vsx based features then enable vsx and
    // altivec. We'll diagnose any problems later.
    bool FeatureHasVSX = llvm::StringSwitch<bool>(Name)
                             .Case("vsx", true)
                             .Case("direct-move", true)
                             .Case("power8-vector", true)
                             .Case("power9-vector", true)
                             .Case("paired-vector-memops", true)
                             .Case("power10-vector", true)
                             .Case("float128", true)
                             .Default(false);
    if (FeatureHasVSX)
      Features["vsx"] = Features["altivec"] = true;
    if (Name == "power9-vector")
      Features["power8-vector"] = true;
    else if (Name == "power10-vector")
      Features["power8-vector"] = Features["power9-vector"] = true;
    if (Name == "pcrel")
      Features["pcrelative-memops"] = true;
    else
      Features[Name] = true;
  } else {
    // If we're disabling altivec or vsx go ahead and disable all of the vsx
    // features.
    if ((Name == "altivec") || (Name == "vsx"))
      Features["vsx"] = Features["direct-move"] = Features["power8-vector"] =
          Features["float128"] = Features["power9-vector"] =
              Features["paired-vector-memops"] = Features["power10-vector"] =
                  false;
    if (Name == "power8-vector")
      Features["power9-vector"] = Features["paired-vector-memops"] =
          Features["power10-vector"] = false;
    else if (Name == "power9-vector")
      Features["paired-vector-memops"] = Features["power10-vector"] = false;
    if (Name == "pcrel")
      Features["pcrelative-memops"] = false;
    else
      Features[Name] = false;
  }
}

const char *const PPCTargetInfo::GCCRegNames[] = {
    "r0",  "r1",     "r2",   "r3",      "r4",      "r5",  "r6",  "r7",  "r8",
    "r9",  "r10",    "r11",  "r12",     "r13",     "r14", "r15", "r16", "r17",
    "r18", "r19",    "r20",  "r21",     "r22",     "r23", "r24", "r25", "r26",
    "r27", "r28",    "r29",  "r30",     "r31",     "f0",  "f1",  "f2",  "f3",
    "f4",  "f5",     "f6",   "f7",      "f8",      "f9",  "f10", "f11", "f12",
    "f13", "f14",    "f15",  "f16",     "f17",     "f18", "f19", "f20", "f21",
    "f22", "f23",    "f24",  "f25",     "f26",     "f27", "f28", "f29", "f30",
    "f31", "mq",     "lr",   "ctr",     "ap",      "cr0", "cr1", "cr2", "cr3",
    "cr4", "cr5",    "cr6",  "cr7",     "xer",     "v0",  "v1",  "v2",  "v3",
    "v4",  "v5",     "v6",   "v7",      "v8",      "v9",  "v10", "v11", "v12",
    "v13", "v14",    "v15",  "v16",     "v17",     "v18", "v19", "v20", "v21",
    "v22", "v23",    "v24",  "v25",     "v26",     "v27", "v28", "v29", "v30",
    "v31", "vrsave", "vscr", "spe_acc", "spefscr", "sfp"
};

ArrayRef<const char *> PPCTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

const TargetInfo::GCCRegAlias PPCTargetInfo::GCCRegAliases[] = {
    // While some of these aliases do map to different registers
    // they still share the same register name.
    {{"0"}, "r0"},     {{"1"}, "r1"},     {{"2"}, "r2"},     {{"3"}, "r3"},
    {{"4"}, "r4"},     {{"5"}, "r5"},     {{"6"}, "r6"},     {{"7"}, "r7"},
    {{"8"}, "r8"},     {{"9"}, "r9"},     {{"10"}, "r10"},   {{"11"}, "r11"},
    {{"12"}, "r12"},   {{"13"}, "r13"},   {{"14"}, "r14"},   {{"15"}, "r15"},
    {{"16"}, "r16"},   {{"17"}, "r17"},   {{"18"}, "r18"},   {{"19"}, "r19"},
    {{"20"}, "r20"},   {{"21"}, "r21"},   {{"22"}, "r22"},   {{"23"}, "r23"},
    {{"24"}, "r24"},   {{"25"}, "r25"},   {{"26"}, "r26"},   {{"27"}, "r27"},
    {{"28"}, "r28"},   {{"29"}, "r29"},   {{"30"}, "r30"},   {{"31"}, "r31"},
    {{"fr0"}, "f0"},   {{"fr1"}, "f1"},   {{"fr2"}, "f2"},   {{"fr3"}, "f3"},
    {{"fr4"}, "f4"},   {{"fr5"}, "f5"},   {{"fr6"}, "f6"},   {{"fr7"}, "f7"},
    {{"fr8"}, "f8"},   {{"fr9"}, "f9"},   {{"fr10"}, "f10"}, {{"fr11"}, "f11"},
    {{"fr12"}, "f12"}, {{"fr13"}, "f13"}, {{"fr14"}, "f14"}, {{"fr15"}, "f15"},
    {{"fr16"}, "f16"}, {{"fr17"}, "f17"}, {{"fr18"}, "f18"}, {{"fr19"}, "f19"},
    {{"fr20"}, "f20"}, {{"fr21"}, "f21"}, {{"fr22"}, "f22"}, {{"fr23"}, "f23"},
    {{"fr24"}, "f24"}, {{"fr25"}, "f25"}, {{"fr26"}, "f26"}, {{"fr27"}, "f27"},
    {{"fr28"}, "f28"}, {{"fr29"}, "f29"}, {{"fr30"}, "f30"}, {{"fr31"}, "f31"},
    {{"cc"}, "cr0"},
};

ArrayRef<TargetInfo::GCCRegAlias> PPCTargetInfo::getGCCRegAliases() const {
  return llvm::makeArrayRef(GCCRegAliases);
}

// PPC ELFABIv2 DWARF Definitoin "Table 2.26. Mappings of Common Registers".
// vs0 ~ vs31 is mapping to 32 - 63,
// vs32 ~ vs63 is mapping to 77 - 108. 
const TargetInfo::AddlRegName GCCAddlRegNames[] = {
    // Table of additional register names to use in user input.
    {{"vs0"}, 32},   {{"vs1"}, 33},   {{"vs2"}, 34},   {{"vs3"}, 35}, 
    {{"vs4"}, 36},   {{"vs5"}, 37},   {{"vs6"}, 38},   {{"vs7"}, 39},
    {{"vs8"}, 40},   {{"vs9"}, 41},   {{"vs10"}, 42},  {{"vs11"}, 43},
    {{"vs12"}, 44},  {{"vs13"}, 45},  {{"vs14"}, 46},  {{"vs15"}, 47},
    {{"vs16"}, 48},  {{"vs17"}, 49},  {{"vs18"}, 50},  {{"vs19"}, 51},
    {{"vs20"}, 52},  {{"vs21"}, 53},  {{"vs22"}, 54},  {{"vs23"}, 55},
    {{"vs24"}, 56},  {{"vs25"}, 57},  {{"vs26"}, 58},  {{"vs27"}, 59},
    {{"vs28"}, 60},  {{"vs29"}, 61},  {{"vs30"}, 62},  {{"vs31"}, 63},
    {{"vs32"}, 77},  {{"vs33"}, 78},  {{"vs34"}, 79},  {{"vs35"}, 80},
    {{"vs36"}, 81},  {{"vs37"}, 82},  {{"vs38"}, 83},  {{"vs39"}, 84},
    {{"vs40"}, 85},  {{"vs41"}, 86},  {{"vs42"}, 87},  {{"vs43"}, 88},
    {{"vs44"}, 89},  {{"vs45"}, 90},  {{"vs46"}, 91},  {{"vs47"}, 92},
    {{"vs48"}, 93},  {{"vs49"}, 94},  {{"vs50"}, 95},  {{"vs51"}, 96},
    {{"vs52"}, 97},  {{"vs53"}, 98},  {{"vs54"}, 99},  {{"vs55"}, 100},
    {{"vs56"}, 101}, {{"vs57"}, 102}, {{"vs58"}, 103}, {{"vs59"}, 104},
    {{"vs60"}, 105}, {{"vs61"}, 106}, {{"vs62"}, 107}, {{"vs63"}, 108},
};

ArrayRef<TargetInfo::AddlRegName> PPCTargetInfo::getGCCAddlRegNames() const {
  if (ABI == "elfv2")
    return llvm::makeArrayRef(GCCAddlRegNames);
  else 
    return TargetInfo::getGCCAddlRegNames(); 
}

static constexpr llvm::StringLiteral ValidCPUNames[] = {
    {"generic"}, {"440"},     {"450"},       {"601"},     {"602"},
    {"603"},     {"603e"},    {"603ev"},     {"604"},     {"604e"},
    {"620"},     {"630"},     {"g3"},        {"7400"},    {"g4"},
    {"7450"},    {"g4+"},     {"750"},       {"8548"},    {"970"},
    {"g5"},      {"a2"},      {"e500"},      {"e500mc"},  {"e5500"},
    {"power3"},  {"pwr3"},    {"power4"},    {"pwr4"},    {"power5"},
    {"pwr5"},    {"power5x"}, {"pwr5x"},     {"power6"},  {"pwr6"},
    {"power6x"}, {"pwr6x"},   {"power7"},    {"pwr7"},    {"power8"},
    {"pwr8"},    {"power9"},  {"pwr9"},      {"power10"}, {"pwr10"},
    {"powerpc"}, {"ppc"},     {"powerpc64"}, {"ppc64"},   {"powerpc64le"},
    {"ppc64le"}, {"future"}};

bool PPCTargetInfo::isValidCPUName(StringRef Name) const {
  return llvm::find(ValidCPUNames, Name) != std::end(ValidCPUNames);
}

void PPCTargetInfo::fillValidCPUList(SmallVectorImpl<StringRef> &Values) const {
  Values.append(std::begin(ValidCPUNames), std::end(ValidCPUNames));
}

void PPCTargetInfo::adjust(LangOptions &Opts) {
  if (HasAltivec)
    Opts.AltiVec = 1;
  TargetInfo::adjust(Opts);
  if (LongDoubleFormat != &llvm::APFloat::IEEEdouble())
    LongDoubleFormat = Opts.PPCIEEELongDouble
                           ? &llvm::APFloat::IEEEquad()
                           : &llvm::APFloat::PPCDoubleDouble();
}

ArrayRef<Builtin::Info> PPCTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::PPC::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}
