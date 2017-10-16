//===--- AMDGPU.cpp - Implement AMDGPU target feature support -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements AMDGPU TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

namespace clang {
namespace targets {

// If you edit the description strings, make sure you update
// getPointerWidthV().

static const char *const DataLayoutStringR600 =
    "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
    "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64";

static const char *const DataLayoutStringSIPrivateIsZero =
    "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32"
    "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
    "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64";

static const char *const DataLayoutStringSIGenericIsZero =
    "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32"
    "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
    "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5";

static const LangASMap AMDGPUPrivIsZeroDefIsGenMap = {
    4, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    0, // opencl_private
    4, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};

static const LangASMap AMDGPUGenIsZeroDefIsGenMap = {
    0, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    5, // opencl_private
    0, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};

static const LangASMap AMDGPUPrivIsZeroDefIsPrivMap = {
    0, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    0, // opencl_private
    4, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};

static const LangASMap AMDGPUGenIsZeroDefIsPrivMap = {
    5, // Default
    1, // opencl_global
    3, // opencl_local
    2, // opencl_constant
    5, // opencl_private
    0, // opencl_generic
    1, // cuda_device
    2, // cuda_constant
    3  // cuda_shared
};
} // namespace targets
} // namespace clang

const Builtin::Info AMDGPUTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, FEATURE},
#include "clang/Basic/BuiltinsAMDGPU.def"
};

const char *const AMDGPUTargetInfo::GCCRegNames[] = {
  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
  "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
  "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
  "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35",
  "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44",
  "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52", "v53",
  "v54", "v55", "v56", "v57", "v58", "v59", "v60", "v61", "v62",
  "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70", "v71",
  "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79", "v80",
  "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89",
  "v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98",
  "v99", "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107",
  "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116",
  "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124", "v125",
  "v126", "v127", "v128", "v129", "v130", "v131", "v132", "v133", "v134",
  "v135", "v136", "v137", "v138", "v139", "v140", "v141", "v142", "v143",
  "v144", "v145", "v146", "v147", "v148", "v149", "v150", "v151", "v152",
  "v153", "v154", "v155", "v156", "v157", "v158", "v159", "v160", "v161",
  "v162", "v163", "v164", "v165", "v166", "v167", "v168", "v169", "v170",
  "v171", "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179",
  "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188",
  "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196", "v197",
  "v198", "v199", "v200", "v201", "v202", "v203", "v204", "v205", "v206",
  "v207", "v208", "v209", "v210", "v211", "v212", "v213", "v214", "v215",
  "v216", "v217", "v218", "v219", "v220", "v221", "v222", "v223", "v224",
  "v225", "v226", "v227", "v228", "v229", "v230", "v231", "v232", "v233",
  "v234", "v235", "v236", "v237", "v238", "v239", "v240", "v241", "v242",
  "v243", "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251",
  "v252", "v253", "v254", "v255", "s0", "s1", "s2", "s3", "s4",
  "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13",
  "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22",
  "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
  "s32", "s33", "s34", "s35", "s36", "s37", "s38", "s39", "s40",
  "s41", "s42", "s43", "s44", "s45", "s46", "s47", "s48", "s49",
  "s50", "s51", "s52", "s53", "s54", "s55", "s56", "s57", "s58",
  "s59", "s60", "s61", "s62", "s63", "s64", "s65", "s66", "s67",
  "s68", "s69", "s70", "s71", "s72", "s73", "s74", "s75", "s76",
  "s77", "s78", "s79", "s80", "s81", "s82", "s83", "s84", "s85",
  "s86", "s87", "s88", "s89", "s90", "s91", "s92", "s93", "s94",
  "s95", "s96", "s97", "s98", "s99", "s100", "s101", "s102", "s103",
  "s104", "s105", "s106", "s107", "s108", "s109", "s110", "s111", "s112",
  "s113", "s114", "s115", "s116", "s117", "s118", "s119", "s120", "s121",
  "s122", "s123", "s124", "s125", "s126", "s127", "exec", "vcc", "scc",
  "m0", "flat_scratch", "exec_lo", "exec_hi", "vcc_lo", "vcc_hi", 
  "flat_scratch_lo", "flat_scratch_hi"
};

ArrayRef<const char *> AMDGPUTargetInfo::getGCCRegNames() const {
  return llvm::makeArrayRef(GCCRegNames);
}

bool AMDGPUTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeatureVec) const {

  // XXX - What does the member GPU mean if device name string passed here?
  if (getTriple().getArch() == llvm::Triple::amdgcn) {
    if (CPU.empty())
      CPU = "tahiti";

    switch (parseAMDGCNName(CPU)) {
    case GK_GFX6:
    case GK_GFX7:
      break;

    case GK_GFX9:
      Features["gfx9-insts"] = true;
      LLVM_FALLTHROUGH;
    case GK_GFX8:
      Features["s-memrealtime"] = true;
      Features["16-bit-insts"] = true;
      Features["dpp"] = true;
      break;

    case GK_NONE:
      return false;
    default:
      llvm_unreachable("unhandled subtarget");
    }
  } else {
    if (CPU.empty())
      CPU = "r600";

    switch (parseR600Name(CPU)) {
    case GK_R600:
    case GK_R700:
    case GK_EVERGREEN:
    case GK_NORTHERN_ISLANDS:
      break;
    case GK_R600_DOUBLE_OPS:
    case GK_R700_DOUBLE_OPS:
    case GK_EVERGREEN_DOUBLE_OPS:
    case GK_CAYMAN:
      Features["fp64"] = true;
      break;
    case GK_NONE:
      return false;
    default:
      llvm_unreachable("unhandled subtarget");
    }
  }

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeatureVec);
}

void AMDGPUTargetInfo::adjustTargetOptions(const CodeGenOptions &CGOpts,
                                           TargetOptions &TargetOpts) const {
  bool hasFP32Denormals = false;
  bool hasFP64Denormals = false;
  for (auto &I : TargetOpts.FeaturesAsWritten) {
    if (I == "+fp32-denormals" || I == "-fp32-denormals")
      hasFP32Denormals = true;
    if (I == "+fp64-fp16-denormals" || I == "-fp64-fp16-denormals")
      hasFP64Denormals = true;
  }
  if (!hasFP32Denormals)
    TargetOpts.Features.push_back(
        (Twine(hasFullSpeedFMAF32(TargetOpts.CPU) && !CGOpts.FlushDenorm
                   ? '+'
                   : '-') +
         Twine("fp32-denormals"))
            .str());
  // Always do not flush fp64 or fp16 denorms.
  if (!hasFP64Denormals && hasFP64)
    TargetOpts.Features.push_back("+fp64-fp16-denormals");
}

AMDGPUTargetInfo::GPUKind AMDGPUTargetInfo::parseR600Name(StringRef Name) {
  return llvm::StringSwitch<GPUKind>(Name)
      .Case("r600", GK_R600)
      .Case("rv610", GK_R600)
      .Case("rv620", GK_R600)
      .Case("rv630", GK_R600)
      .Case("rv635", GK_R600)
      .Case("rs780", GK_R600)
      .Case("rs880", GK_R600)
      .Case("rv670", GK_R600_DOUBLE_OPS)
      .Case("rv710", GK_R700)
      .Case("rv730", GK_R700)
      .Case("rv740", GK_R700_DOUBLE_OPS)
      .Case("rv770", GK_R700_DOUBLE_OPS)
      .Case("palm", GK_EVERGREEN)
      .Case("cedar", GK_EVERGREEN)
      .Case("sumo", GK_EVERGREEN)
      .Case("sumo2", GK_EVERGREEN)
      .Case("redwood", GK_EVERGREEN)
      .Case("juniper", GK_EVERGREEN)
      .Case("hemlock", GK_EVERGREEN_DOUBLE_OPS)
      .Case("cypress", GK_EVERGREEN_DOUBLE_OPS)
      .Case("barts", GK_NORTHERN_ISLANDS)
      .Case("turks", GK_NORTHERN_ISLANDS)
      .Case("caicos", GK_NORTHERN_ISLANDS)
      .Case("cayman", GK_CAYMAN)
      .Case("aruba", GK_CAYMAN)
      .Default(GK_NONE);
}

AMDGPUTargetInfo::GPUKind AMDGPUTargetInfo::parseAMDGCNName(StringRef Name) {
  return llvm::StringSwitch<GPUKind>(Name)
      .Case("gfx600", GK_GFX6)
      .Case("tahiti", GK_GFX6)
      .Case("gfx601", GK_GFX6)
      .Case("pitcairn", GK_GFX6)
      .Case("verde", GK_GFX6)
      .Case("oland", GK_GFX6)
      .Case("hainan", GK_GFX6)
      .Case("gfx700", GK_GFX7)
      .Case("bonaire", GK_GFX7)
      .Case("kaveri", GK_GFX7)
      .Case("gfx701", GK_GFX7)
      .Case("hawaii", GK_GFX7)
      .Case("gfx702", GK_GFX7)
      .Case("gfx703", GK_GFX7)
      .Case("kabini", GK_GFX7)
      .Case("mullins", GK_GFX7)
      .Case("gfx800", GK_GFX8)
      .Case("iceland", GK_GFX8)
      .Case("gfx801", GK_GFX8)
      .Case("carrizo", GK_GFX8)
      .Case("gfx802", GK_GFX8)
      .Case("tonga", GK_GFX8)
      .Case("gfx803", GK_GFX8)
      .Case("fiji", GK_GFX8)
      .Case("polaris10", GK_GFX8)
      .Case("polaris11", GK_GFX8)
      .Case("gfx804", GK_GFX8)
      .Case("gfx810", GK_GFX8)
      .Case("stoney", GK_GFX8)
      .Case("gfx900", GK_GFX9)
      .Case("gfx901", GK_GFX9)
      .Case("gfx902", GK_GFX9)
      .Case("gfx903", GK_GFX9)
      .Default(GK_NONE);
}

void AMDGPUTargetInfo::setAddressSpaceMap(bool DefaultIsPrivate) {
  if (isGenericZero(getTriple())) {
    AddrSpaceMap = DefaultIsPrivate ? &AMDGPUGenIsZeroDefIsPrivMap
                                    : &AMDGPUGenIsZeroDefIsGenMap;
  } else {
    AddrSpaceMap = DefaultIsPrivate ? &AMDGPUPrivIsZeroDefIsPrivMap
                                    : &AMDGPUPrivIsZeroDefIsGenMap;
  }
}

AMDGPUTargetInfo::AMDGPUTargetInfo(const llvm::Triple &Triple,
                                   const TargetOptions &Opts)
    : TargetInfo(Triple), GPU(isAMDGCN(Triple) ? GK_GFX6 : GK_R600),
      hasFP64(false), hasFMAF(false), hasLDEXPF(false),
      AS(isGenericZero(Triple)) {
  if (getTriple().getArch() == llvm::Triple::amdgcn) {
    hasFP64 = true;
    hasFMAF = true;
    hasLDEXPF = true;
  }
  auto IsGenericZero = isGenericZero(Triple);
  resetDataLayout(getTriple().getArch() == llvm::Triple::amdgcn
                      ? (IsGenericZero ? DataLayoutStringSIGenericIsZero
                                       : DataLayoutStringSIPrivateIsZero)
                      : DataLayoutStringR600);
  assert(DataLayout->getAllocaAddrSpace() == AS.Private);

  setAddressSpaceMap(Triple.getOS() == llvm::Triple::Mesa3D ||
                     Triple.getEnvironment() == llvm::Triple::OpenCL ||
                     Triple.getEnvironmentName() == "amdgizcl" ||
                     !isAMDGCN(Triple));
  UseAddrSpaceMapMangling = true;

  // Set pointer width and alignment for target address space 0.
  PointerWidth = PointerAlign = DataLayout->getPointerSizeInBits();
  if (getMaxPointerWidth() == 64) {
    LongWidth = LongAlign = 64;
    SizeType = UnsignedLong;
    PtrDiffType = SignedLong;
    IntPtrType = SignedLong;
  }

  MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
}

void AMDGPUTargetInfo::adjust(LangOptions &Opts) {
  TargetInfo::adjust(Opts);
  setAddressSpaceMap(Opts.OpenCL || !isAMDGCN(getTriple()));
}

ArrayRef<Builtin::Info> AMDGPUTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::AMDGPU::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}

void AMDGPUTargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  if (getTriple().getArch() == llvm::Triple::amdgcn)
    Builder.defineMacro("__AMDGCN__");
  else
    Builder.defineMacro("__R600__");

  if (hasFMAF)
    Builder.defineMacro("__HAS_FMAF__");
  if (hasLDEXPF)
    Builder.defineMacro("__HAS_LDEXPF__");
  if (hasFP64)
    Builder.defineMacro("__HAS_FP64__");
}
