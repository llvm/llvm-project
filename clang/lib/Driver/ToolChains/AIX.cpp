//===--- AIX.cpp - AIX ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIX.h"
#include "Arch/PPC.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Path.h"

using AIX = clang::driver::toolchains::AIX;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;

using namespace llvm::opt;
using namespace llvm::sys;

void aix::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  const bool IsArch32Bit = getToolChain().getTriple().isArch32Bit();
  const bool IsArch64Bit = getToolChain().getTriple().isArch64Bit();
  // Only support 32 and 64 bit.
  if (!IsArch32Bit && !IsArch64Bit)
    llvm_unreachable("Unsupported bit width value.");

  // Specify the mode in which the as(1) command operates.
  if (IsArch32Bit) {
    CmdArgs.push_back("-a32");
  } else {
    // Must be 64-bit, otherwise asserted already.
    CmdArgs.push_back("-a64");
  }

  // Accept any mixture of instructions.
  // On Power for AIX and Linux, this behaviour matches that of GCC for both the
  // user-provided assembler source case and the compiler-produced assembler
  // source case. Yet XL with user-provided assembler source would not add this.
  CmdArgs.push_back("-many");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  // Specify assembler output file.
  assert((Output.isFilename() || Output.isNothing()) && "Invalid output.");
  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }

  // Specify assembler input file.
  // The system assembler on AIX takes exactly one input file. The driver is
  // expected to invoke as(1) separately for each assembler source input file.
  if (Inputs.size() != 1)
    llvm_unreachable("Invalid number of input files.");
  const InputInfo &II = Inputs[0];
  assert((II.isFilename() || II.isNothing()) && "Invalid input.");
  if (II.isFilename())
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Inputs, Output));
}

// Determine whether there are any linker options that supply an export list
// (or equivalent information about what to export) being sent to the linker.
static bool hasExportListLinkerOpts(const ArgStringList &CmdArgs) {
  for (size_t i = 0, Size = CmdArgs.size(); i < Size; ++i) {
    llvm::StringRef ArgString(CmdArgs[i]);

    if (ArgString.startswith("-bE:") || ArgString.startswith("-bexport:") ||
        ArgString == "-bexpall" || ArgString == "-bexpfull")
      return true;

    // If we split -b option, check the next opt.
    if (ArgString == "-b" && i + 1 < Size) {
      ++i;
      llvm::StringRef ArgNextString(CmdArgs[i]);
      if (ArgNextString.startswith("E:") ||
          ArgNextString.startswith("export:") || ArgNextString == "expall" ||
          ArgNextString == "expfull")
        return true;
    }
  }
  return false;
}

void aix::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                               const InputInfo &Output,
                               const InputInfoList &Inputs, const ArgList &Args,
                               const char *LinkingOutput) const {
  const AIX &ToolChain = static_cast<const AIX &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;

  const bool IsArch32Bit = ToolChain.getTriple().isArch32Bit();
  const bool IsArch64Bit = ToolChain.getTriple().isArch64Bit();
  // Only support 32 and 64 bit.
  if (!(IsArch32Bit || IsArch64Bit))
    llvm_unreachable("Unsupported bit width value.");

  // Force static linking when "-static" is present.
  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-bnso");

  // Add options for shared libraries.
  if (Args.hasArg(options::OPT_shared)) {
    CmdArgs.push_back("-bM:SRE");
    CmdArgs.push_back("-bnoentry");
  }

  // PGO instrumentation generates symbols belonging to special sections, and
  // the linker needs to place all symbols in a particular section together in
  // memory; the AIX linker does that under an option.
  if (Args.hasFlag(options::OPT_fprofile_arcs, options::OPT_fno_profile_arcs,
                    false) ||
       Args.hasFlag(options::OPT_fprofile_generate,
                    options::OPT_fno_profile_generate, false) ||
       Args.hasFlag(options::OPT_fprofile_generate_EQ,
                    options::OPT_fno_profile_generate, false) ||
       Args.hasFlag(options::OPT_fprofile_instr_generate,
                    options::OPT_fno_profile_instr_generate, false) ||
       Args.hasFlag(options::OPT_fprofile_instr_generate_EQ,
                    options::OPT_fno_profile_instr_generate, false) ||
       Args.hasFlag(options::OPT_fcs_profile_generate,
                    options::OPT_fno_profile_generate, false) ||
       Args.hasFlag(options::OPT_fcs_profile_generate_EQ,
                    options::OPT_fno_profile_generate, false) ||
       Args.hasArg(options::OPT_fcreate_profile) ||
       Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back("-bdbg:namedsects:ss");

  // Specify linker output file.
  assert((Output.isFilename() || Output.isNothing()) && "Invalid output.");
  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }

  // Set linking mode (i.e., 32/64-bit) and the address of
  // text and data sections based on arch bit width.
  if (IsArch32Bit) {
    CmdArgs.push_back("-b32");
    CmdArgs.push_back("-bpT:0x10000000");
    CmdArgs.push_back("-bpD:0x20000000");
  } else {
    // Must be 64-bit, otherwise asserted already.
    CmdArgs.push_back("-b64");
    CmdArgs.push_back("-bpT:0x100000000");
    CmdArgs.push_back("-bpD:0x110000000");
  }

  auto getCrt0Basename = [&Args, IsArch32Bit] {
    // Enable gprofiling when "-pg" is specified.
    if (Args.hasArg(options::OPT_pg))
      return IsArch32Bit ? "gcrt0.o" : "gcrt0_64.o";
    // Enable profiling when "-p" is specified.
    else if (Args.hasArg(options::OPT_p))
      return IsArch32Bit ? "mcrt0.o" : "mcrt0_64.o";
    else
      return IsArch32Bit ? "crt0.o" : "crt0_64.o";
  };

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_shared)) {
    CmdArgs.push_back(
        Args.MakeArgString(ToolChain.GetFilePath(getCrt0Basename())));

    CmdArgs.push_back(Args.MakeArgString(
        ToolChain.GetFilePath(IsArch32Bit ? "crti.o" : "crti_64.o")));
  }

  // Collect all static constructor and destructor functions in both C and CXX
  // language link invocations. This has to come before AddLinkerInputs as the
  // implied option needs to precede any other '-bcdtors' settings or
  // '-bnocdtors' that '-Wl' might forward.
  CmdArgs.push_back("-bcdtors:all:0:s");

  // Specify linker input file(s).
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  if (D.isUsingLTO()) {
    assert(!Inputs.empty() && "Must have at least one input.");
    addLTOOptions(ToolChain, Args, CmdArgs, Output, Inputs[0],
                  D.getLTOMode() == LTOK_Thin);
  }

  if (Args.hasArg(options::OPT_shared) && !hasExportListLinkerOpts(CmdArgs)) {

    const char *CreateExportListExec = Args.MakeArgString(
        path::parent_path(ToolChain.getDriver().ClangExecutable) +
        "/llvm-nm");
    ArgStringList CreateExportCmdArgs;

    std::string CreateExportListPath =
        C.getDriver().GetTemporaryPath("CreateExportList", "exp");
    const char *ExportList =
        C.addTempFile(C.getArgs().MakeArgString(CreateExportListPath));

    for (const auto &II : Inputs)
      if (II.isFilename())
        CreateExportCmdArgs.push_back(II.getFilename());

    CreateExportCmdArgs.push_back("--export-symbols");
    CreateExportCmdArgs.push_back("-X");
    if (IsArch32Bit) {
      CreateExportCmdArgs.push_back("32");
    } else {
      // Must be 64-bit, otherwise asserted already.
      CreateExportCmdArgs.push_back("64");
    }

    auto ExpCommand = std::make_unique<Command>(
        JA, *this, ResponseFileSupport::None(), CreateExportListExec,
        CreateExportCmdArgs, Inputs, Output);
    ExpCommand->setRedirectFiles(
        {std::nullopt, std::string(ExportList), std::nullopt});
    C.addCommand(std::move(ExpCommand));
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("-bE:") + ExportList));
  }

  // Add directory to library search path.
  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  ToolChain.addProfileRTLibs(Args, CmdArgs);

  if (getToolChain().ShouldLinkCXXStdlib(Args))
    getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

    // Add OpenMP runtime if -fopenmp is specified.
    if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                     options::OPT_fno_openmp, false)) {
      switch (ToolChain.getDriver().getOpenMPRuntime(Args)) {
      case Driver::OMPRT_OMP:
        CmdArgs.push_back("-lomp");
        break;
      case Driver::OMPRT_IOMP5:
        CmdArgs.push_back("-liomp5");
        break;
      case Driver::OMPRT_GOMP:
        CmdArgs.push_back("-lgomp");
        break;
      case Driver::OMPRT_Unknown:
        // Already diagnosed.
        break;
      }
    }

    // Support POSIX threads if "-pthreads" or "-pthread" is present.
    if (Args.hasArg(options::OPT_pthreads, options::OPT_pthread))
      CmdArgs.push_back("-lpthreads");

    if (D.CCCIsCXX())
      CmdArgs.push_back("-lm");

    CmdArgs.push_back("-lc");

    if (Args.hasArg(options::OPT_pg)) {
      CmdArgs.push_back(Args.MakeArgString((llvm::Twine("-L") + D.SysRoot) +
                                           "/lib/profiled"));
      CmdArgs.push_back(Args.MakeArgString((llvm::Twine("-L") + D.SysRoot) +
                                           "/usr/lib/profiled"));
    }
  }

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                         Exec, CmdArgs, Inputs, Output));
}

/// AIX - AIX tool chain which can call as(1) and ld(1) directly.
AIX::AIX(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  ParseInlineAsmUsingAsmParser = Args.hasFlag(
      options::OPT_fintegrated_as, options::OPT_fno_integrated_as, true);
  getLibraryPaths().push_back(getDriver().SysRoot + "/usr/lib");
}

// Returns the effective header sysroot path to use.
// This comes from either -isysroot or --sysroot.
llvm::StringRef
AIX::GetHeaderSysroot(const llvm::opt::ArgList &DriverArgs) const {
  if (DriverArgs.hasArg(options::OPT_isysroot))
    return DriverArgs.getLastArgValue(options::OPT_isysroot);
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;
  return "/";
}

void AIX::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                    ArgStringList &CC1Args) const {
  // Return if -nostdinc is specified as a driver option.
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  llvm::StringRef Sysroot = GetHeaderSysroot(DriverArgs);
  const Driver &D = getDriver();

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    // Add the PowerPC intrinsic headers (<resource>/include/ppc_wrappers)
    path::append(P, "include", "ppc_wrappers");
    addSystemInclude(DriverArgs, CC1Args, P);
    // Add the Clang builtin headers (<resource>/include)
    addSystemInclude(DriverArgs, CC1Args, path::parent_path(P.str()));
  }

  // Return if -nostdlibinc is specified as a driver option.
  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Add <sysroot>/usr/include.
  SmallString<128> UP(Sysroot);
  path::append(UP, "/usr/include");
  addSystemInclude(DriverArgs, CC1Args, UP.str());
}

void AIX::AddClangCXXStdlibIncludeArgs(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {

  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libstdcxx:
    llvm::report_fatal_error(
        "picking up libstdc++ headers is unimplemented on AIX");
  case ToolChain::CST_Libcxx: {
    llvm::StringRef Sysroot = GetHeaderSysroot(DriverArgs);
    SmallString<128> PathCPP(Sysroot);
    llvm::sys::path::append(PathCPP, "opt/IBM/openxlCSDK", "include", "c++",
                            "v1");
    addSystemInclude(DriverArgs, CC1Args, PathCPP.str());
    // Required in order to suppress conflicting C++ overloads in the system
    // libc headers that were used by XL C++.
    CC1Args.push_back("-D__LIBC_NO_CPP_MATH_OVERLOADS__");
    return;
  }
  }

  llvm_unreachable("Unexpected C++ library type; only libc++ is supported.");
}

void AIX::AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                              llvm::opt::ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libstdcxx:
    llvm::report_fatal_error("linking libstdc++ unimplemented on AIX");
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    if (Args.hasArg(options::OPT_fexperimental_library))
      CmdArgs.push_back("-lc++experimental");
    CmdArgs.push_back("-lc++abi");
    return;
  }

  llvm_unreachable("Unexpected C++ library type; only libc++ is supported.");
}

void AIX::addProfileRTLibs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const {
  // Add linker option -u__llvm_profile_runtime to cause runtime
  // initialization to occur.
  if (needsProfileRT(Args))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("-u", llvm::getInstrProfRuntimeHookVarName())));
  ToolChain::addProfileRTLibs(Args, CmdArgs);
}

ToolChain::CXXStdlibType AIX::GetDefaultCXXStdlibType() const {
  return ToolChain::CST_Libcxx;
}

ToolChain::RuntimeLibType AIX::GetDefaultRuntimeLibType() const {
  return ToolChain::RLT_CompilerRT;
}

auto AIX::buildAssembler() const -> Tool * { return new aix::Assembler(*this); }

auto AIX::buildLinker() const -> Tool * { return new aix::Linker(*this); }
