//===--- AmdOptArgs.cpp - Args handling for multiple toolchains -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Closed optimization compiler is invoked if -famd-opt is specified, or
// if any of the closed optimization flags are specified on the command line.
// These can also include -mllvm options as well as -f<options>
//===----------------------------------------------------------------------===//

#include "CommonArgs.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static bool hasLlvmAoccOption(const ArgList &Args) {
  llvm::StringMap<bool> Flags;
  Flags.insert(std::make_pair("-enable-X86-prefetching", true));
  Flags.insert(std::make_pair("-enable-licm-vrp", true));
  Flags.insert(std::make_pair("-function-specialize", true));
  Flags.insert(std::make_pair("-enable-nans-for-sqrt", true));
  Flags.insert(std::make_pair("-phi-elim-preserve-cmpjmp-glue", true));
  Flags.insert(std::make_pair("-delay-vectorization-to-lto", true));
  Flags.insert(std::make_pair("-enable-partial-unswitch", true));
  Flags.insert(std::make_pair("-aggressive-loop-unswitch", true));
  Flags.insert(std::make_pair("-convert-pow-exp-to-int", true));
  Flags.insert(std::make_pair("-global-vectorize-slp", true));
  Flags.insert(std::make_pair("-move-load-slice-gslp", true));
  Flags.insert(std::make_pair("-reduce-array-computations", true));
  Flags.insert(std::make_pair("-remap-arrays", true));
  Flags.insert(std::make_pair("-struct-peel-mem-block-size", true));
  Flags.insert(std::make_pair("-lv-function-specialization", true));
  Flags.insert(std::make_pair("-disable-itodcalls", true));
  Flags.insert(std::make_pair("-disable-itodcallsbyclone", true));
  Flags.insert(std::make_pair("-rv-boscc", true));
  Flags.insert(std::make_pair("-region-vectorize", true));
  Flags.insert(std::make_pair("-mark-rv-outline", true));
  Flags.insert(std::make_pair("-rv-outline", true));
  Flags.insert(std::make_pair("-rv-depth", true));
  Flags.insert(std::make_pair("-rv-max-reg-size", true));
  Flags.insert(std::make_pair("-enable-branch-combine", true));
  Flags.insert(std::make_pair("-simplifycfg-no-storesink", true));
  Flags.insert(std::make_pair("-inline-aggressive", true));

  for (Arg *A : Args) {
    if (!A->getNumValues()) continue;
    std::string S(A->getValue(0));
    if (Flags.count(S))
      return true;
  }
  return false;
}

static bool hasOption(const ArgList &Args, const char *opt) {
   for (auto begin = Args.begin(), end = Args.end(); begin != end; begin++) {
       if ((*begin)->containsValue(opt)) {
          return true;
       }
   }
   return false;
}

static void addCmdArgs(const ArgList &Args, ArgStringList &CmdArgs,
                       bool isLLD, bool checkOnly, const char *Arg,
		       bool noPrefix=false) {
  if (checkOnly)
    return;
  if (isLLD) {
    const Twine Str = "-plugin-opt=";
    CmdArgs.push_back(Args.MakeArgString(Str + Arg));
  } else if (noPrefix) {
    CmdArgs.push_back(Arg);
  } else if (!isLLD) {
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(Arg);
  } else {
    // Nothing, isLLD and !isLLD make this dead patch
  }
}

static bool checkForPropOpts(const ToolChain &TC, const Driver &D,
                             const ArgList &Args, ArgStringList &CmdArgs,
                             bool isLLD, bool checkOnly, bool HasAltPath) {
  bool OFastEnabled = isOptimizationLevelFast(Args);
  bool ClosedToolChainNeeded = hasLlvmAoccOption(Args);

  // Enable -loop-unswitch-aggressive opt flag, only when
  // 1) -Ofast
  // 2) -floop-unswitch-aggressive
  if (((ClosedToolChainNeeded && OFastEnabled && HasAltPath &&
        !Args.hasArg(options::OPT_fno_loop_unswitch_aggressive)) ||
       Args.hasArg(options::OPT_floop_unswitch_aggressive)) &&
      !hasOption(Args, Args.MakeArgString("-aggressive-loop-unswitch"))) {
    if (!checkOnly) {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back("-aggressive-loop-unswitch");
    }
    ClosedToolChainNeeded = true;
  }

  if (Args.hasFlag(options::OPT_finline_aggressive,
                   options::OPT_fno_inline_aggressive, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-inline-aggressive");
    ClosedToolChainNeeded = true;
  }

  if (Arg *A = Args.getLastArg(options::OPT_fnt_store_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-nt-store=" + Val));
    ClosedToolChainNeeded = true;
  }

  if (Arg *A = Args.getLastArg(options::OPT_fveclib)) {
    StringRef Name = A->getValue();
    if ((Name == "Accelerate") || (Name == "none") ||
       (Name == "MASSV") || (Name == "SVML") ||
        (Name == "AMDLIBM"))
      addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                 Args.MakeArgString("-vector-library=" + Name));
    else if (( Name == "libmvec")) {
      switch(TC.getTriple().getArch()) {
      default:
        break;
      case llvm::Triple::x86_64:
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                   Args.MakeArgString("-vector-library=LIBMVEC-X86"));
        break;
      }
    }
    // fveclib supported prior to amd-opt, if its AMDLIBM then
    // we want to trigger closed compiler, otherwise not.
    if (Name == "AMDLIBM")
      ClosedToolChainNeeded = true;
  }

  if (Arg *A = Args.getLastArg(options::OPT_fstruct_layout_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-struct-layout=" + Val), isLLD);
    ClosedToolChainNeeded = true;
  }

  if (Args.hasFlag(options::OPT_fremove_unused_array_ops,
                   options::OPT_fnoremove_unused_array_ops, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-remove-unused-array-ops"));
    ClosedToolChainNeeded = true;
  }

  if (Arg *A = Args.getLastArg(options::OPT_finline_recursion_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-inline-recursion=" + Val));
    ClosedToolChainNeeded = true;
  }
  if (Args.hasFlag(options::OPT_farray_remap, options::OPT_fno_array_remap,
                   false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-remap-arrays", isLLD);
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-simplifycfg-no-storesink");
    ClosedToolChainNeeded = true;
  }

  if (Arg *A = Args.getLastArg(options::OPT_fstruct_peel_ptr_size_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-struct-peel-ptr-size=" + Val));
    ClosedToolChainNeeded = true;
  }

  if (Arg *A = Args.getLastArg(options::OPT_Rpass_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-pass-remarks=" + Val));
    // RPass supported prior to famd-opt, so not a trigger.
  }

  if (Arg *A = Args.getLastArg(options::OPT_Rpass_missed_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-pass-remarks-missed=" + Val));
    // RPass_missed supported prior to famd-opt, so not a trigger.
  }

  if (Arg *A = Args.getLastArg(options::OPT_Rpass_analysis_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-pass-remarks-analysis=" + Val));
    // RPass_analysis supported prior to famd-opt, so not a trigger.
  }

  if (Args.hasFlag(options::OPT_fsimplify_pow, options::OPT_fno_simplify_pow,
                   false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-simplify-pow");
    ClosedToolChainNeeded = true;
  }

  if (Args.hasFlag(options::OPT_floop_splitting,
                   options::OPT_fno_loop_splitting, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-loop-splitting");
    ClosedToolChainNeeded = true;
  }

  if (Args.hasFlag(options::OPT_fno_loop_splitting,
                   options::OPT_floop_splitting, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-loop-splitting=false");
    ClosedToolChainNeeded = true;
  }

  if (Args.hasFlag(options::OPT_fproactive_loop_fusion_analysis,
                   options::OPT_fno_proactive_loop_fusion_analysis, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               "-proactive-loop-fusion-analysis");
    ClosedToolChainNeeded = true;
  }

  if (Args.hasFlag(options::OPT_fproactive_loop_fusion,
                   options::OPT_fno_proactive_loop_fusion, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-proactive-loop-fusion");
    ClosedToolChainNeeded = true;
  }

  // Screwball logic warning: the following code keys off the march
  // and we only want to add these closed options if we are already closed.
  if (Arg *A = Args.getLastArg(options::OPT_fstruct_peel_mem_block_size_EQ)) {
    StringRef Val = A->getValue();
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               Args.MakeArgString("-struct-peel-mem-block-size=" + Val),
	       isLLD);
    ClosedToolChainNeeded = true;
  } else if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    if (ClosedToolChainNeeded && HasAltPath) {
      std::string CPU = getCPUName(D, Args, TC.getTriple());
      StringRef MArch = A->getValue();
#define ZNVER1_MEMBLOCK_SIZE "8192"
#define ZNVER2_MEMBLOCK_SIZE "16384"
      if (MArch == "znver1" || (MArch == "native" && CPU == "znver1")) {
        // Tune mem-block-szie for znver1
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                   Args.MakeArgString(
                       "-struct-peel-mem-block-size=" ZNVER1_MEMBLOCK_SIZE),
		   isLLD);
        ClosedToolChainNeeded = true;
      }
      if (MArch == "znver2" || (MArch == "native" && CPU == "znver2")) {
        // Tune mem-block-size for znver2
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                   Args.MakeArgString(
                       "-struct-peel-mem-block-size=" ZNVER2_MEMBLOCK_SIZE),
		   isLLD);
        ClosedToolChainNeeded = true;
      }
      if (MArch == "znver3" || (MArch == "native" && CPU == "znver3")) {
        // Tune mem-block-size for znver3
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                   Args.MakeArgString(
                       "-struct-peel-mem-block-size=" ZNVER2_MEMBLOCK_SIZE),
		   isLLD);
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                   Args.MakeArgString("-x86-convert-cmpmr-to-cmprm"));
        ClosedToolChainNeeded = true;
      }
    }
  }
  if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    if (ClosedToolChainNeeded && HasAltPath) {
      StringRef MArch = A->getValue();
      if (MArch == "znver1") {
      if (!checkOnly) {
        CmdArgs.push_back("-mllvm");
        CmdArgs.push_back("-slp-max-reg-size-def=128");
      }
        ClosedToolChainNeeded = true;
      } else if ((MArch == "znver2") || (MArch == "znver3")) {
        // -rv-max-reg-size=256 around 5% gain on nab
      if (!checkOnly) {
        CmdArgs.push_back("-mllvm");
        CmdArgs.push_back("-rv-max-reg-size=256");
      }
        ClosedToolChainNeeded = true;
      }
    }
  }
  if (Args.hasFlag(options::OPT_fno_branch_combine,
                   options::OPT_fno_branch_combine, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-enable-branch-combine=false");
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               "-phi-elim-preserve-cmpjmp-glue=false");
    ClosedToolChainNeeded = true;
  } else if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    if (ClosedToolChainNeeded && HasAltPath) {
      StringRef MArch = A->getValue();
      if (MArch == "znver1" || MArch == "znver2" || MArch == "znver3") {
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-enable-branch-combine");
        addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                   "-phi-elim-preserve-cmpjmp-glue");
        ClosedToolChainNeeded = true;
      }
    }
  }
  if (Args.hasFlag(options::OPT_flv_function_specialization,
                   options::OPT_fno_lv_function_specialization, false)) {
    addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
               "-lv-function-specialization", isLLD);
    if (!isLLD)
      addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-delay-vectorization-to-lto");
    ClosedToolChainNeeded = true;
  }

  if (ClosedToolChainNeeded && isLLD) {
    if (Args.hasFlag(options::OPT_fno_itodcalls, options::OPT_fitodcalls,
                     false)) {
      addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-disable-itodcalls");
    } else {
      addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-disable-itodcalls=false");
    }

    if (Args.hasFlag(options::OPT_fno_itodcallsbyclone,
                     options::OPT_fitodcallsbyclone, false)) {
      addCmdArgs(Args, CmdArgs, isLLD, checkOnly, "-disable-itodcallsbyclone");
    } else {
      addCmdArgs(Args, CmdArgs, isLLD, checkOnly,
                 "-disable-itodcallsbyclone=false");
    }
  }

  return ClosedToolChainNeeded;
}

bool tools::checkForAMDProprietaryOptOptions(
    const ToolChain &TC, const Driver &D, const ArgList &Args,
    ArgStringList &CmdArgs, bool isLLD, bool checkOnly) {

  bool ProprietaryToolChainNeeded = false;
  std::string AltPath = D.getInstalledDir();
  AltPath += "/../alt/bin";
  // -famd-opt enables prorietary compiler and lto
  if (Args.hasFlag(options::OPT_famd_opt, options::OPT_fno_amd_opt, false)) {
    if (!TC.getVFS().exists(AltPath)) {
      D.Diag(diag::warn_drv_amd_opt_not_found);
      return false;
    }
    ProprietaryToolChainNeeded = true;
  }
  // disables amd proprietary compiler
  if (Args.hasFlag(options::OPT_fno_amd_opt, options::OPT_famd_opt, false)) {
    return false;
  }

  // check for more AOCC options
  ProprietaryToolChainNeeded |= checkForPropOpts(
      TC, D, Args, CmdArgs, isLLD, checkOnly, TC.getVFS().exists(AltPath));

  if (ProprietaryToolChainNeeded && !TC.getVFS().exists(AltPath)) {
    D.Diag(diag::warn_drv_amd_opt_not_found);
    return false;
  }
  return ProprietaryToolChainNeeded;
}
