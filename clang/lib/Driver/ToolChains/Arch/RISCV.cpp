//===--- RISCV.cpp - RISCV Helpers for Tools --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/raw_ostream.h"
#include "ToolChains/CommonArgs.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

namespace {
// Represents the major and version number components of a RISC-V extension
struct RISCVExtensionVersion {
  StringRef Major;
  StringRef Minor;
};
} // end anonymous namespace

static StringRef getExtensionTypeDesc(StringRef Ext) {
  if (Ext.startswith("sx"))
    return "non-standard supervisor-level extension";
  if (Ext.startswith("s"))
    return "standard supervisor-level extension";
  if (Ext.startswith("x"))
    return "non-standard user-level extension";
  if (Ext.startswith("z"))
    return "standard user-level extension";
  return StringRef();
}

static StringRef getExtensionType(StringRef Ext) {
  if (Ext.startswith("sx"))
    return "sx";
  if (Ext.startswith("s"))
    return "s";
  if (Ext.startswith("x"))
    return "x";
  if (Ext.startswith("z"))
    return "z";
  return StringRef();
}

// If the extension is supported as experimental, return the version of that
// extension that the compiler currently supports.
static Optional<RISCVExtensionVersion>
isExperimentalExtension(StringRef Ext) {
  if (Ext == "b" || Ext == "zba" || Ext == "zbb" || Ext == "zbc" ||
      Ext == "zbe" || Ext == "zbf" || Ext == "zbm" || Ext == "zbp" ||
      Ext == "zbr" || Ext == "zbs" || Ext == "zbt" || Ext == "zbproposedc")
    return RISCVExtensionVersion{"0", "93"};
  if (Ext == "v" || Ext == "zvamo" || Ext == "zvlsseg")
    return RISCVExtensionVersion{"0", "10"};
  if (Ext == "zfh")
    return RISCVExtensionVersion{"0", "1"};
  return None;
}

static bool isSupportedExtension(StringRef Ext) {
  // LLVM supports "z" extensions which are marked as experimental.
  if (isExperimentalExtension(Ext))
    return true;

  // LLVM does not support "sx", "s" nor "x" extensions.
  return false;
}

// Extensions may have a version number, and may be separated by
// an underscore '_' e.g.: rv32i2_m2.
// Version number is divided into major and minor version numbers,
// separated by a 'p'. If the minor version is 0 then 'p0' can be
// omitted from the version string. E.g., rv32i2p0, rv32i2, rv32i2p1.
static bool getExtensionVersion(const Driver &D, const ArgList &Args,
                                StringRef MArch, StringRef Ext, StringRef In,
                                std::string &Major, std::string &Minor) {
  Major = std::string(In.take_while(isDigit));
  In = In.substr(Major.size());

  if (Major.size() && In.consume_front("p")) {
    Minor = std::string(In.take_while(isDigit));
    In = In.substr(Major.size() + 1);

    // Expected 'p' to be followed by minor version number.
    if (Minor.empty()) {
      std::string Error =
        "minor version number missing after 'p' for extension";
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
        << MArch << Error << Ext;
      return false;
    }
  }

  // Expected multi-character extension with version number to have no
  // subsequent characters (i.e. must either end string or be followed by
  // an underscore).
  if (Ext.size() > 1 && In.size()) {
    std::string Error =
        "multi-character extensions must be separated by underscores";
    D.Diag(diag::err_drv_invalid_riscv_ext_arch_name) << MArch << Error << In;
    return false;
  }

  // If experimental extension, require use of current version number number
  if (auto ExperimentalExtension = isExperimentalExtension(Ext)) {
    if (!Args.hasArg(options::OPT_menable_experimental_extensions)) {
      std::string Error =
          "requires '-menable-experimental-extensions' for experimental extension";
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
          << MArch << Error << Ext;
      return false;
    } else if (Major.empty() && Minor.empty()) {
      std::string Error =
          "experimental extension requires explicit version number";
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
          << MArch << Error << Ext;
      return false;
    }
    auto SupportedVers = *ExperimentalExtension;
    if (Major != SupportedVers.Major || Minor != SupportedVers.Minor) {
      std::string Error =
          "unsupported version number " + Major;
      if (!Minor.empty())
        Error += "." + Minor;
      Error += " for experimental extension (this compiler supports "
            + SupportedVers.Major.str() + "."
            + SupportedVers.Minor.str() + ")";

      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
          << MArch << Error << Ext;
      return false;
    }
    return true;
  }

  // Allow extensions to declare no version number
  if (Major.empty() && Minor.empty())
    return true;

  // TODO: Handle supported extensions with version number.
  std::string Error = "unsupported version number " + Major;
  if (!Minor.empty())
    Error += "." + Minor;
  Error += " for extension";
  D.Diag(diag::err_drv_invalid_riscv_ext_arch_name) << MArch << Error << Ext;

  return false;
}

// Handle other types of extensions other than the standard
// general purpose and standard user-level extensions.
// Parse the ISA string containing non-standard user-level
// extensions, standard supervisor-level extensions and
// non-standard supervisor-level extensions.
// These extensions start with 'z', 'x', 's', 'sx' prefixes, follow a
// canonical order, might have a version number (major, minor)
// and are separated by a single underscore '_'.
// Set the hardware features for the extensions that are supported.
static void getExtensionFeatures(const Driver &D,
                                 const ArgList &Args,
                                 std::vector<StringRef> &Features,
                                 StringRef &MArch, StringRef &Exts) {
  if (Exts.empty())
    return;

  // Multi-letter extensions are seperated by a single underscore
  // as described in RISC-V User-Level ISA V2.2.
  SmallVector<StringRef, 8> Split;
  Exts.split(Split, StringRef("_"));

  SmallVector<StringRef, 4> Prefix{"z", "x", "s", "sx"};
  auto I = Prefix.begin();
  auto E = Prefix.end();

  SmallVector<StringRef, 8> AllExts;

  for (StringRef Ext : Split) {
    if (Ext.empty()) {
      D.Diag(diag::err_drv_invalid_riscv_arch_name) << MArch
        << "extension name missing after separator '_'";
      return;
    }

    StringRef Type = getExtensionType(Ext);
    StringRef Desc = getExtensionTypeDesc(Ext);
    auto Pos = Ext.find_if(isDigit);
    StringRef Name(Ext.substr(0, Pos));
    StringRef Vers(Ext.substr(Pos));

    if (Type.empty()) {
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
        << MArch << "invalid extension prefix" << Ext;
      return;
    }

    // Check ISA extensions are specified in the canonical order.
    while (I != E && *I != Type)
      ++I;

    if (I == E) {
      std::string Error = std::string(Desc);
      Error += " not given in canonical order";
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
        << MArch <<  Error << Ext;
      return;
    }

    // The order is OK, do not advance I to the next prefix
    // to allow repeated extension type, e.g.: rv32ixabc_xdef.

    if (Name.size() == Type.size()) {
      std::string Error = std::string(Desc);
      Error += " name missing after";
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
        << MArch << Error << Type;
      return;
    }

    std::string Major, Minor;
    if (!getExtensionVersion(D, Args, MArch, Name, Vers, Major, Minor))
      return;

    // Check if duplicated extension.
    if (llvm::is_contained(AllExts, Name)) {
      std::string Error = "duplicated ";
      Error += Desc;
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
        << MArch << Error << Name;
      return;
    }

    // Extension format is correct, keep parsing the extensions.
    // TODO: Save Type, Name, Major, Minor to avoid parsing them later.
    AllExts.push_back(Name);
  }

  // Set target features.
  // TODO: Hardware features to be handled in Support/TargetParser.cpp.
  // TODO: Use version number when setting target features.
  for (auto Ext : AllExts) {
    if (!isSupportedExtension(Ext)) {
      StringRef Desc = getExtensionTypeDesc(getExtensionType(Ext));
      std::string Error = "unsupported ";
      Error += Desc;
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
        << MArch << Error << Ext;
      return;
    }
    if (Ext == "zvamo" || Ext == "zvlsseg") {
      Features.push_back("+experimental-v");
      Features.push_back("+experimental-zvamo");
      Features.push_back("+experimental-zvlsseg");
    } else if (isExperimentalExtension(Ext))
      Features.push_back(Args.MakeArgString("+experimental-" + Ext));
    else
      Features.push_back(Args.MakeArgString("+" + Ext));
  }
}

// Returns false if an error is diagnosed.
static bool getArchFeatures(const Driver &D, StringRef MArch,
                            std::vector<StringRef> &Features,
                            const ArgList &Args) {
  // RISC-V ISA strings must be lowercase.
  if (llvm::any_of(MArch, [](char c) { return isupper(c); })) {
    D.Diag(diag::err_drv_invalid_riscv_arch_name)
        << MArch << "string must be lowercase";
    return false;
  }

  // ISA string must begin with rv32 or rv64.
  if (!(MArch.startswith("rv32") || MArch.startswith("rv64")) ||
      (MArch.size() < 5)) {
    D.Diag(diag::err_drv_invalid_riscv_arch_name)
        << MArch << "string must begin with rv32{i,e,g} or rv64{i,g}";
    return false;
  }

  bool HasRV64 = MArch.startswith("rv64");

  // The canonical order specified in ISA manual.
  // Ref: Table 22.1 in RISC-V User-Level ISA V2.2
  StringRef StdExts = "mafdqlcbjtpvn";
  bool HasF = false, HasD = false;
  char Baseline = MArch[4];

  // First letter should be 'e', 'i' or 'g'.
  switch (Baseline) {
  default:
    D.Diag(diag::err_drv_invalid_riscv_arch_name)
        << MArch << "first letter should be 'e', 'i' or 'g'";
    return false;
  case 'e': {
    StringRef Error;
    // Currently LLVM does not support 'e'.
    // Extension 'e' is not allowed in rv64.
    if (HasRV64)
      Error = "standard user-level extension 'e' requires 'rv32'";
    else
      Error = "unsupported standard user-level extension 'e'";
    D.Diag(diag::err_drv_invalid_riscv_arch_name) << MArch << Error;
    return false;
  }
  case 'i':
    break;
  case 'g':
    // g = imafd
    StdExts = StdExts.drop_front(4);
    Features.push_back("+m");
    Features.push_back("+a");
    Features.push_back("+f");
    Features.push_back("+d");
    HasF = true;
    HasD = true;
    break;
  }

  // Skip rvxxx
  StringRef Exts = MArch.substr(5);

  // Remove multi-letter standard extensions, non-standard extensions and
  // supervisor-level extensions. They have 'z', 'x', 's', 'sx' prefixes.
  // Parse them at the end.
  // Find the very first occurrence of 's', 'x' or 'z'.
  StringRef OtherExts;
  size_t Pos = Exts.find_first_of("zsx");
  if (Pos != StringRef::npos) {
    OtherExts = Exts.substr(Pos);
    Exts = Exts.substr(0, Pos);
  }

  std::string Major, Minor;
  if (!getExtensionVersion(D, Args, MArch, std::string(1, Baseline), Exts,
                           Major, Minor))
    return false;

  // Consume the base ISA version number and any '_' between rvxxx and the
  // first extension
  Exts = Exts.drop_front(Major.size());
  if (!Minor.empty())
    Exts = Exts.drop_front(Minor.size() + 1 /*'p'*/);
  Exts.consume_front("_");

  // TODO: Use version number when setting target features

  auto StdExtsItr = StdExts.begin();
  auto StdExtsEnd = StdExts.end();

  for (auto I = Exts.begin(), E = Exts.end(); I != E; ) {
    char c = *I;

    // Check ISA extensions are specified in the canonical order.
    while (StdExtsItr != StdExtsEnd && *StdExtsItr != c)
      ++StdExtsItr;

    if (StdExtsItr == StdExtsEnd) {
      // Either c contains a valid extension but it was not given in
      // canonical order or it is an invalid extension.
      StringRef Error;
      if (StdExts.contains(c))
        Error = "standard user-level extension not given in canonical order";
      else
        Error = "invalid standard user-level extension";
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
          << MArch << Error << std::string(1, c);
      return false;
    }

    // Move to next char to prevent repeated letter.
    ++StdExtsItr;

    std::string Next, Major, Minor;
    if (std::next(I) != E)
      Next = std::string(std::next(I), E);
    if (!getExtensionVersion(D, Args, MArch, std::string(1, c), Next, Major,
                             Minor))
      return false;

    // The order is OK, then push it into features.
    // TODO: Use version number when setting target features
    switch (c) {
    default:
      // Currently LLVM supports only "mafdc".
      D.Diag(diag::err_drv_invalid_riscv_ext_arch_name)
          << MArch << "unsupported standard user-level extension"
          << std::string(1, c);
      return false;
    case 'm':
      Features.push_back("+m");
      break;
    case 'a':
      Features.push_back("+a");
      break;
    case 'f':
      Features.push_back("+f");
      HasF = true;
      break;
    case 'd':
      Features.push_back("+d");
      HasD = true;
      break;
    case 'c':
      Features.push_back("+c");
      break;
    case 'b':
      Features.push_back("+experimental-b");
      Features.push_back("+experimental-zba");
      Features.push_back("+experimental-zbb");
      Features.push_back("+experimental-zbc");
      Features.push_back("+experimental-zbe");
      Features.push_back("+experimental-zbf");
      Features.push_back("+experimental-zbm");
      Features.push_back("+experimental-zbp");
      Features.push_back("+experimental-zbr");
      Features.push_back("+experimental-zbs");
      Features.push_back("+experimental-zbt");
      break;
    case 'v':
      Features.push_back("+experimental-v");
      Features.push_back("+experimental-zvamo");
      Features.push_back("+experimental-zvlsseg");
      break;
    }

    // Consume full extension name and version, including any optional '_'
    // between this extension and the next
    ++I;
    I += Major.size();
    if (Minor.size())
      I += Minor.size() + 1 /*'p'*/;
    if (*I == '_')
      ++I;
  }

  // Dependency check.
  // It's illegal to specify the 'd' (double-precision floating point)
  // extension without also specifying the 'f' (single precision
  // floating-point) extension.
  if (HasD && !HasF) {
    D.Diag(diag::err_drv_invalid_riscv_arch_name)
        << MArch << "d requires f extension to also be specified";
    return false;
  }

  // Additional dependency checks.
  // TODO: The 'q' extension requires rv64.
  // TODO: It is illegal to specify 'e' extensions with 'f' and 'd'.

  // Handle all other types of extensions.
  getExtensionFeatures(D, Args, Features, MArch, OtherExts);

  return true;
}

// Get features except standard extension feature
static void getRISCFeaturesFromMcpu(const Driver &D, const llvm::Triple &Triple,
                                    const llvm::opt::ArgList &Args,
                                    const llvm::opt::Arg *A, StringRef Mcpu,
                                    std::vector<StringRef> &Features) {
  bool Is64Bit = (Triple.getArch() == llvm::Triple::riscv64);
  llvm::RISCV::CPUKind CPUKind = llvm::RISCV::parseCPUKind(Mcpu);
  if (!llvm::RISCV::checkCPUKind(CPUKind, Is64Bit) ||
      !llvm::RISCV::getCPUFeaturesExceptStdExt(CPUKind, Features)) {
    D.Diag(clang::diag::err_drv_clang_unsupported) << A->getAsString(Args);
  }
}

void riscv::getRISCVTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                   const ArgList &Args,
                                   std::vector<StringRef> &Features) {
  StringRef MArch = getRISCVArch(Args, Triple);

  if (!getArchFeatures(D, MArch, Features, Args))
    return;

  // If users give march and mcpu, get std extension feature from MArch
  // and other features (ex. mirco architecture feature) from mcpu
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    getRISCFeaturesFromMcpu(D, Triple, Args, A, A->getValue(), Features);

  // Handle features corresponding to "-ffixed-X" options
  if (Args.hasArg(options::OPT_ffixed_x1))
    Features.push_back("+reserve-x1");
  if (Args.hasArg(options::OPT_ffixed_x2))
    Features.push_back("+reserve-x2");
  if (Args.hasArg(options::OPT_ffixed_x3))
    Features.push_back("+reserve-x3");
  if (Args.hasArg(options::OPT_ffixed_x4))
    Features.push_back("+reserve-x4");
  if (Args.hasArg(options::OPT_ffixed_x5))
    Features.push_back("+reserve-x5");
  if (Args.hasArg(options::OPT_ffixed_x6))
    Features.push_back("+reserve-x6");
  if (Args.hasArg(options::OPT_ffixed_x7))
    Features.push_back("+reserve-x7");
  if (Args.hasArg(options::OPT_ffixed_x8))
    Features.push_back("+reserve-x8");
  if (Args.hasArg(options::OPT_ffixed_x9))
    Features.push_back("+reserve-x9");
  if (Args.hasArg(options::OPT_ffixed_x10))
    Features.push_back("+reserve-x10");
  if (Args.hasArg(options::OPT_ffixed_x11))
    Features.push_back("+reserve-x11");
  if (Args.hasArg(options::OPT_ffixed_x12))
    Features.push_back("+reserve-x12");
  if (Args.hasArg(options::OPT_ffixed_x13))
    Features.push_back("+reserve-x13");
  if (Args.hasArg(options::OPT_ffixed_x14))
    Features.push_back("+reserve-x14");
  if (Args.hasArg(options::OPT_ffixed_x15))
    Features.push_back("+reserve-x15");
  if (Args.hasArg(options::OPT_ffixed_x16))
    Features.push_back("+reserve-x16");
  if (Args.hasArg(options::OPT_ffixed_x17))
    Features.push_back("+reserve-x17");
  if (Args.hasArg(options::OPT_ffixed_x18))
    Features.push_back("+reserve-x18");
  if (Args.hasArg(options::OPT_ffixed_x19))
    Features.push_back("+reserve-x19");
  if (Args.hasArg(options::OPT_ffixed_x20))
    Features.push_back("+reserve-x20");
  if (Args.hasArg(options::OPT_ffixed_x21))
    Features.push_back("+reserve-x21");
  if (Args.hasArg(options::OPT_ffixed_x22))
    Features.push_back("+reserve-x22");
  if (Args.hasArg(options::OPT_ffixed_x23))
    Features.push_back("+reserve-x23");
  if (Args.hasArg(options::OPT_ffixed_x24))
    Features.push_back("+reserve-x24");
  if (Args.hasArg(options::OPT_ffixed_x25))
    Features.push_back("+reserve-x25");
  if (Args.hasArg(options::OPT_ffixed_x26))
    Features.push_back("+reserve-x26");
  if (Args.hasArg(options::OPT_ffixed_x27))
    Features.push_back("+reserve-x27");
  if (Args.hasArg(options::OPT_ffixed_x28))
    Features.push_back("+reserve-x28");
  if (Args.hasArg(options::OPT_ffixed_x29))
    Features.push_back("+reserve-x29");
  if (Args.hasArg(options::OPT_ffixed_x30))
    Features.push_back("+reserve-x30");
  if (Args.hasArg(options::OPT_ffixed_x31))
    Features.push_back("+reserve-x31");

  // -mrelax is default, unless -mno-relax is specified.
  if (Args.hasFlag(options::OPT_mrelax, options::OPT_mno_relax, true))
    Features.push_back("+relax");
  else
    Features.push_back("-relax");

  // GCC Compatibility: -mno-save-restore is default, unless -msave-restore is
  // specified.
  if (Args.hasFlag(options::OPT_msave_restore, options::OPT_mno_save_restore, false))
    Features.push_back("+save-restore");
  else
    Features.push_back("-save-restore");

  // Now add any that the user explicitly requested on the command line,
  // which may override the defaults.
  handleTargetFeaturesGroup(Args, Features, options::OPT_m_riscv_Features_Group);
}

StringRef riscv::getRISCVABI(const ArgList &Args, const llvm::Triple &Triple) {
  assert((Triple.getArch() == llvm::Triple::riscv32 ||
          Triple.getArch() == llvm::Triple::riscv64) &&
         "Unexpected triple");

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
  // rv64* -> lp64
  StringRef MArch = getRISCVArch(Args, Triple);

  if (MArch.startswith_lower("rv32")) {
    // FIXME: parse `March` to find `D` extension properly
    if (MArch.substr(4).contains_lower("d") || MArch.startswith_lower("rv32g"))
      return "ilp32d";
    else if (MArch.startswith_lower("rv32e"))
      return "ilp32e";
    else
      return "ilp32";
  } else if (MArch.startswith_lower("rv64")) {
    // FIXME: parse `March` to find `D` extension properly
    if (MArch.substr(4).contains_lower("d") || MArch.startswith_lower("rv64g"))
      return "lp64d";
    else
      return "lp64";
  }

  // 3. Choose a default based on the triple
  //
  // We deviate from GCC's defaults here:
  // - On `riscv{XLEN}-unknown-elf` we use the integer calling convention only.
  // - On all other OSs we use the double floating point calling convention.
  if (Triple.getArch() == llvm::Triple::riscv32) {
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

StringRef riscv::getRISCVArch(const llvm::opt::ArgList &Args,
                              const llvm::Triple &Triple) {
  assert((Triple.getArch() == llvm::Triple::riscv32 ||
          Triple.getArch() == llvm::Triple::riscv64) &&
         "Unexpected triple");

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
  // 1. Explicit choices using `-march=`
  // 2. Based on `-mcpu` if the target CPU has a default ISA string
  // 3. A default based on `-mabi`, if provided
  // 4. A default based on the target triple's arch
  //
  // Clang does not yet support MULTILIB_REUSE, so we use `rv{XLEN}imafdc`
  // instead of `rv{XLEN}gc` though they are (currently) equivalent.

  // 1. If `-march=` is specified, use it.
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
    return A->getValue();

  // 2. Get march (isa string) based on `-mcpu=`
  if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    StringRef MArch = llvm::RISCV::getMArchFromMcpu(A->getValue());
    // Bypass if target cpu's default march is empty.
    if (MArch != "")
      return MArch;
  }

  // 3. Choose a default based on `-mabi=`
  //
  // ilp32e -> rv32e
  // ilp32 | ilp32f | ilp32d -> rv32imafdc
  // lp64 | lp64f | lp64d -> rv64imafdc
  if (const Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    StringRef MABI = A->getValue();

    if (MABI.equals_lower("ilp32e"))
      return "rv32e";
    else if (MABI.startswith_lower("ilp32"))
      return "rv32imafdc";
    else if (MABI.startswith_lower("lp64"))
      return "rv64imafdc";
  }

  // 4. Choose a default based on the triple
  //
  // We deviate from GCC's defaults here:
  // - On `riscv{XLEN}-unknown-elf` we default to `rv{XLEN}imac`
  // - On all other OSs we use `rv{XLEN}imafdc` (equivalent to `rv{XLEN}gc`)
  if (Triple.getArch() == llvm::Triple::riscv32) {
    if (Triple.getOS() == llvm::Triple::UnknownOS)
      return "rv32imac";
    else
      return "rv32imafdc";
  } else {
    if (Triple.getOS() == llvm::Triple::UnknownOS)
      return "rv64imac";
    else
      return "rv64imafdc";
  }
}
