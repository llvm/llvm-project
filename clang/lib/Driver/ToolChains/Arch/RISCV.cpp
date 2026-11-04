//===--- RISCV.cpp - RISC-V Helpers for Tools -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "../Clang.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Options/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

// Returns false if an error is diagnosed.
static bool getArchFeatures(const Driver &D, StringRef Arch,
                            std::vector<StringRef> &Features,
                            const ArgList &Args) {
  bool EnableExperimentalExtensions =
      Args.hasArg(options::OPT_menable_experimental_extensions);
  auto ISAInfo =
      llvm::RISCVISAInfo::parseArchString(Arch, EnableExperimentalExtensions);
  if (!ISAInfo) {
    handleAllErrors(ISAInfo.takeError(), [&](llvm::StringError &ErrMsg) {
      D.Diag(diag::err_drv_invalid_riscv_arch_name)
          << Arch << ErrMsg.getMessage();
    });

    return false;
  }

  for (const std::string &Str : (*ISAInfo)->toFeatures(/*AddAllExtension=*/true,
                                                       /*IgnoreUnknown=*/false))
    Features.push_back(Args.MakeArgString(Str));

  if (EnableExperimentalExtensions)
    Features.push_back(Args.MakeArgString("+experimental"));

  return true;
}

static bool isValidRISCVCPU(const Driver &D, const Arg *A,
                            const llvm::Triple &Triple, StringRef Mcpu) {
  bool Is64Bit = Triple.isRISCV64();
  if (!llvm::RISCV::parseCPU(Mcpu, Is64Bit)) {
    // Try inverting Is64Bit in case the CPU is valid, but for the wrong target.
    if (llvm::RISCV::parseCPU(Mcpu, !Is64Bit))
      D.Diag(clang::diag::err_drv_invalid_riscv_cpu_name_for_target)
          << Mcpu << Is64Bit;
    else
      D.Diag(clang::diag::err_drv_unsupported_option_argument)
          << A->getSpelling() << Mcpu;
    return false;
  }
  return true;
}

void riscv::getRISCVTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                   const ArgList &Args,
                                   std::vector<StringRef> &Features) {
  std::string MArch = getRISCVArch(Args, Triple);

  if (!getArchFeatures(D, MArch, Features, Args))
    return;

  bool CPUFastScalarUnaligned = false;
  bool CPUFastVectorUnaligned = false;

  StringRef CPU;
  Arg *CPUArg = nullptr;

  if ((CPUArg = Args.getLastArg(options::OPT_mcpu_EQ))) {
    CPU = CPUArg->getValue();
  } else if ((CPUArg = Args.getLastArg(options::OPT_march_EQ))) {
    StringRef MArchValue = CPUArg->getValue();
    if (MArchValue == "native")
      CPU = "native";
  }

  if (!CPU.empty()) {
    if (CPU == "native")
      CPU = llvm::sys::getHostCPUName();

    if (!isValidRISCVCPU(D, CPUArg, Triple, CPU))
      return;

    if (llvm::RISCV::hasFastScalarUnalignedAccess(CPU))
      CPUFastScalarUnaligned = true;
    if (llvm::RISCV::hasFastVectorUnalignedAccess(CPU))
      CPUFastVectorUnaligned = true;
  }

// Handle features corresponding to "-ffixed-X" options
#define RESERVE_REG(REG)                                                       \
  if (Args.hasArg(options::OPT_ffixed_##REG))                                  \
    Features.push_back("+reserve-" #REG);
  RESERVE_REG(x1)
  RESERVE_REG(x2)
  RESERVE_REG(x3)
  RESERVE_REG(x4)
  RESERVE_REG(x5)
  RESERVE_REG(x6)
  RESERVE_REG(x7)
  RESERVE_REG(x8)
  RESERVE_REG(x9)
  RESERVE_REG(x10)
  RESERVE_REG(x11)
  RESERVE_REG(x12)
  RESERVE_REG(x13)
  RESERVE_REG(x14)
  RESERVE_REG(x15)
  RESERVE_REG(x16)
  RESERVE_REG(x17)
  RESERVE_REG(x18)
  RESERVE_REG(x19)
  RESERVE_REG(x20)
  RESERVE_REG(x21)
  RESERVE_REG(x22)
  RESERVE_REG(x23)
  RESERVE_REG(x24)
  RESERVE_REG(x25)
  RESERVE_REG(x26)
  RESERVE_REG(x27)
  RESERVE_REG(x28)
  RESERVE_REG(x29)
  RESERVE_REG(x30)
  RESERVE_REG(x31)
#undef RESERVE_REG

  // -mrelax is default, unless -mno-relax is specified.
  if (Args.hasFlag(options::OPT_mrelax, options::OPT_mno_relax, true))
    Features.push_back("+relax");
  else
    Features.push_back("-relax");

  // If -mstrict-align, -mno-strict-align, -mscalar-strict-align, or
  // -mno-scalar-strict-align is passed, use it. Otherwise, the
  // unaligned-scalar-mem is enabled if the CPU supports it or the target is
  // Android.
  if (const Arg *A = Args.getLastArg(
          options::OPT_mno_strict_align, options::OPT_mscalar_strict_align,
          options::OPT_mstrict_align, options::OPT_mno_scalar_strict_align)) {
    if (A->getOption().matches(options::OPT_mno_strict_align) ||
        A->getOption().matches(options::OPT_mno_scalar_strict_align)) {
      Features.push_back("+unaligned-scalar-mem");
    } else {
      Features.push_back("-unaligned-scalar-mem");
    }
  } else if (CPUFastScalarUnaligned || Triple.isAndroid()) {
    Features.push_back("+unaligned-scalar-mem");
  }

  // If -mstrict-align, -mno-strict-align, -mvector-strict-align, or
  // -mno-vector-strict-align is passed, use it. Otherwise, the
  // unaligned-vector-mem is enabled if the CPU supports it or the target is
  // Android.
  if (const Arg *A = Args.getLastArg(
          options::OPT_mno_strict_align, options::OPT_mvector_strict_align,
          options::OPT_mstrict_align, options::OPT_mno_vector_strict_align)) {
    if (A->getOption().matches(options::OPT_mno_strict_align) ||
        A->getOption().matches(options::OPT_mno_vector_strict_align)) {
      Features.push_back("+unaligned-vector-mem");
    } else {
      Features.push_back("-unaligned-vector-mem");
    }
  } else if (CPUFastVectorUnaligned || Triple.isAndroid()) {
    Features.push_back("+unaligned-vector-mem");
  }

  if (Triple.isRISCV32()) {
    // Handle `-mzilsd-word-align` and `-mzilsd-strict-align` on rv32. These
    // interact with the scalar alignment options - if unaligned scalar memory
    // is allowed then that takes precedence over this option, as zilsd accesses
    // can be 1-byte aligned in this case. Otherwise, the option
    // `-mzilsd-word-align` option allows zilsd accesses to be 4-byte aligned
    // rather than the usual 8-byte aligned (`-mzilsd-strict-align`).
    if (const Arg *A = Args.getLastArg(
            options::OPT_mstrict_align, options::OPT_mscalar_strict_align,
            options::OPT_mzilsd_word_align, options::OPT_mno_strict_align,
            options::OPT_mno_scalar_strict_align,
            options::OPT_mzilsd_strict_align)) {
      if (A->getOption().matches(options::OPT_mno_strict_align) ||
          A->getOption().matches(options::OPT_mno_scalar_strict_align) ||
          A->getOption().matches(options::OPT_mzilsd_word_align)) {
        Features.push_back("+zilsd-word-align");
      } else {
        Features.push_back("-zilsd-word-align");
      }
    }
  } else {
    // Zilsd is not available on RV64, so report an error for these options.
    if (const Arg *A = Args.getLastArg(options::OPT_mzilsd_word_align,
                                       options::OPT_mzilsd_strict_align)) {
      D.Diag(clang::diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << Triple.getTriple();
    }
  }

  SmallVector<std::string, 4> TuneFeatures;
  if (!riscv::getRISCVTuneCPU(D, Args, &TuneFeatures))
    return;
  for (const std::string &TF : TuneFeatures)
    Features.push_back(Args.MakeArgString(TF));

  // Now add any that the user explicitly requested on the command line,
  // which may override the defaults.
  handleTargetFeaturesGroup(D, Triple, Args, Features,
                            options::OPT_m_riscv_Features_Group);
}

StringRef riscv::getRISCVABI(const ArgList &Args, const llvm::Triple &Triple) {
  assert(Triple.isRISCV() && "Unexpected triple");

  // GCC's logic around choosing a default `-mabi=` is complex. If GCC is not
  // configured using `--with-abi=`, then the logic for the default choice is
  // defined in config.gcc. This function is based on the logic in GCC 9.2.0.
  //
  // The logic used in GCC 9.2.0 is the following, in order:
  // 1. Explicit choices using `--with-abi=`
  // 2. A default based on `--with-arch=`, if provided
  // 3. A default based on the target triple's arch
  //
  // The logic in config.gcc is a little circular but it is not inconsistent.
  //
  // Clang does not have `--with-arch=` or `--with-abi=`, so we use `-march=`
  // and `-mabi=` respectively instead.
  //
  // In order to make chosing logic more clear, Clang uses the following logic,
  // in order:
  // 1. Explicit choices using `-mabi=`
  // 2. A default based on the architecture as determined by getRISCVArch
  // 3. Choose a default based on the triple

  // 1. If `-mabi=` is specified, use it.
  if (const Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    return A->getValue();

  // 2. Choose a default based on the target architecture.
  //
  // rv32g | rv32*d -> ilp32d
  // rv32e -> ilp32e
  // rv32* -> ilp32
  // rv64g | rv64*d -> lp64d
  // rv64e -> lp64e
  // rv64* -> lp64
  std::string Arch = getRISCVArch(Args, Triple);

  auto ParseResult = llvm::RISCVISAInfo::parseArchString(
      Arch, /* EnableExperimentalExtension */ true);
  // Ignore parsing error, just go 3rd step.
  if (!llvm::errorToBool(ParseResult.takeError()))
    return (*ParseResult)->computeDefaultABI();

  // 3. Choose a default based on the triple
  //
  // We deviate from GCC's defaults here:
  // - On `riscv{XLEN}-unknown-elf` we use the integer calling convention only.
  // - On all other OSs we use the double floating point calling convention.
  if (Triple.isRISCV32()) {
    if (Triple.getOS() == llvm::Triple::UnknownOS)
      return "ilp32";
    else
      return "ilp32d";
  } else {
    if (Triple.getOS() == llvm::Triple::UnknownOS)
      return "lp64";
    else
      return "lp64d";
  }
}

static std::string getMArchFromMcpu(StringRef CPU, const llvm::Triple &Triple) {
  if (CPU == "native") {
    CPU = llvm::sys::getHostCPUName();
    // If the target cpu is unrecognized, use target features.
    if (CPU.starts_with("generic")) {
      auto FeatureMap = llvm::sys::getHostCPUFeatures();
      // hwprobe may be unavailable on older Linux versions.
      if (!FeatureMap.empty()) {
        std::vector<std::string> Features;
        for (auto &F : FeatureMap)
          Features.push_back(((F.second ? "+" : "-") + F.first()).str());
        auto ParseResult = llvm::RISCVISAInfo::parseFeatures(
            Triple.isRISCV32() ? 32 : 64, Features);
        if (ParseResult)
          return (*ParseResult)->toString();
      }
    }
  }

  return llvm::RISCV::getMArchFromMcpu(CPU).str();
}

std::string riscv::getRISCVArch(const llvm::opt::ArgList &Args,
                                const llvm::Triple &Triple) {
  assert(Triple.isRISCV() && "Unexpected triple");

  // GCC's logic around choosing a default `-march=` is complex. If GCC is not
  // configured using `--with-arch=`, then the logic for the default choice is
  // defined in config.gcc. This function is based on the logic in GCC 9.2.0. We
  // deviate from GCC's default on additional `-mcpu` option (GCC does not
  // support `-mcpu`) and baremetal targets (UnknownOS) where neither `-march`
  // nor `-mabi` is specified.
  //
  // The logic used in GCC 9.2.0 is the following, in order:
  // 1. Explicit choices using `--with-arch=`
  // 2. A default based on `--with-abi=`, if provided
  // 3. A default based on the target triple's arch
  //
  // The logic in config.gcc is a little circular but it is not inconsistent.
  //
  // Clang does not have `--with-arch=` or `--with-abi=`, so we use `-march=`
  // and `-mabi=` respectively instead.
  //
  // Clang uses the following logic, in order:
  // 1. Explicit choices using `-march=` (`-march=native` means the host CPU)
  // 2. Based on `-mcpu` if `-march=` is not specified and the target CPU has a
  //    default ISA string
  // 3. A default based on `-mabi`, if provided
  // 4. A default based on the target triple's arch
  //
  // Clang does not yet support MULTILIB_REUSE, so we use `rv{XLEN}imafdc`
  // instead of `rv{XLEN}gc` though they are (currently) equivalent.

  // 1. If `-march=` is specified, use it unless the value is "unset". A value
  // of "native" is an alias for `-mcpu=native` and selects the ISA string of
  // the host CPU.
  bool HasMArch = false;
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    StringRef MArchValue = A->getValue();
    if (MArchValue != "unset") {
      HasMArch = true;
      if (MArchValue != "native")
        return MArchValue.str();

      std::string MArch = getMArchFromMcpu(MArchValue, Triple);
      if (!MArch.empty())
        return MArch;
    }
  }

  // 2. Get march (isa string) based on `-mcpu=`. This is only used if `-march=`
  // was not specified, so a `-march=native` that failed to determine the host
  // ISA string above does not fall back to `-mcpu=`.
  if (!HasMArch) {
    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
      std::string MArch = getMArchFromMcpu(A->getValue(), Triple);
      // Bypass if target cpu's default march is empty.
      if (!MArch.empty())
        return MArch;
    }
  }

  // 3. Choose a default based on `-mabi=`
  //
  // ilp32e -> rv32e
  // lp64e -> rv64e
  // ilp32 | ilp32f | ilp32d -> rv32imafdc
  // lp64 | lp64f | lp64d -> rv64imafdc
  if (const Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    StringRef MABI = A->getValue();

    if (MABI.equals_insensitive("ilp32e"))
      return "rv32e";
    if (MABI.equals_insensitive("lp64e"))
      return "rv64e";
    if (MABI.starts_with_insensitive("ilp32"))
      return "rv32imafdc";
    if (MABI.starts_with_insensitive("lp64")) {
      if (Triple.isAndroid())
        return "rv64imafdcv_zba_zbb_zbs";
      if (Triple.isOSFuchsia())
        return "rva22u64_v";
      return "rv64imafdc";
    }
  }

  // 4. Choose a default based on the triple
  //
  // We deviate from GCC's defaults here:
  // - On `riscv{XLEN}-unknown-elf` we default to `rv{XLEN}imac`
  // - On all other OSs we use `rv{XLEN}imafdc` (equivalent to `rv{XLEN}gc`)
  if (Triple.isRISCV32()) {
    if (Triple.getOS() == llvm::Triple::UnknownOS)
      return "rv32imac";
    return "rv32imafdc";
  }

  if (Triple.getOS() == llvm::Triple::UnknownOS)
    return "rv64imac";
  if (Triple.isAndroid())
    return "rv64imafdcv_zba_zbb_zbs";
  if (Triple.isOSFuchsia())
    return "rva22u64_v";
  return "rv64imafdc";
}

std::string riscv::getRISCVTargetCPU(const llvm::opt::ArgList &Args,
                                     const llvm::Triple &Triple) {
  std::string CPU;
  // If we have -mcpu, use that. Otherwise, check for -march=native.
  if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    CPU = A->getValue();
  } else if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    // `-march=native` is an alias for `-mcpu=native`.
    StringRef MArchValue = A->getValue();
    if (MArchValue == "native")
      CPU = "native";
  }

  // Handle CPU name is 'native'.
  if (CPU == "native")
    CPU = llvm::sys::getHostCPUName();

  if (!CPU.empty())
    return CPU;

  return Triple.isRISCV64() ? "generic-rv64" : "generic-rv32";
}

std::optional<StringRef>
riscv::getRISCVTuneCPU(const Driver &D, const llvm::opt::ArgList &Args,
                       SmallVectorImpl<std::string> *TuneFeatures) {
  const Arg *MTuneArg = Args.getLastArg(options::OPT_mtune_EQ);
  if (!MTuneArg)
    return "";

  StringRef TuneCPU = MTuneArg->getValue();
  StringRef TFString;

  auto Idx = TuneCPU.find(':');
  if (Idx != StringRef::npos) {
    if (!Args.hasFlag(options::OPT_mexperimental_mtune_syntax,
                      options::OPT_mno_experimental_mtune_syntax, false)) {
      // Only print this diagnostics if it's used for retrieving tune features
      // to avoid printing the same error message multiple times.
      if (TuneFeatures)
        D.Diag(diag::err_drv_invalid_riscv_mtune_string)
            << 0 << TuneCPU
            << "require '-mexperimental-mtune-syntax' to use with tune feature "
               "string";
      return std::nullopt;
    }

    TFString = TuneCPU.substr(Idx + 1);
    TuneCPU = TuneCPU.slice(0, Idx);
  }

  if (TuneFeatures && !TFString.empty()) {
    if (auto E = llvm::RISCV::parseTuneFeatureString(TuneCPU, TFString,
                                                     *TuneFeatures)) {
      D.Diag(diag::err_drv_invalid_riscv_mtune_string)
          << 1 << TFString << llvm::toString(std::move(E));
      return std::nullopt;
    }
  }

  // Apply -mtune=native after applying features. Not all features apply to
  // all CPUs so an -mtune=native:<feature> may fail depending on what the
  // native was expanded to.
  if (TuneCPU == "native")
    TuneCPU = llvm::sys::getHostCPUName();

  return TuneCPU;
}
