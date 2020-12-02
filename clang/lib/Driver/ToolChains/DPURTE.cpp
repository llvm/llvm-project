//===--- DPURTE.h - DPU RTE ToolChain Implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPURTE.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"

#include "DPUCharacteristics.h"

using namespace llvm::opt;

namespace clang {
namespace driver {
namespace toolchains {

namespace dpu {
void addDPUTargetOptions(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) {
  // add debug information by default if -g0 is not specified
  Arg *A = Args.getLastArg(options::OPT_g_Group);
  if (!A || !A->getOption().matches(options::OPT_g0)) {
    CmdArgs.push_back("-debug-info-kind=limited");
    CmdArgs.push_back("-dwarf-version=4");
  }
}
} // namespace dpu

void DPURTE::AddClangSystemIncludeArgs(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  CC1Args.push_back("-nostdsysteminc");
  addSystemInclude(DriverArgs, CC1Args, StringRef(PathToStdlibIncludes));
  addSystemInclude(DriverArgs, CC1Args, StringRef(PathToSyslibIncludes));
}

char *DPURTE::GetUpmemSdkPath(const char *Path) {
  char *result;
  if (PathToSDK != NULL) {
    asprintf(&result, "%s%s", PathToSDK, Path);
    return result;
  }
  const std::string SysRoot(getDriver().SysRoot);
  const std::string InstalledDir(getDriver().getInstalledDir());
  const std::string UpmemDir(InstalledDir + "/../share/upmem");
  if (!SysRoot.empty()) {
    PathToSDK = strdup(SysRoot.c_str());
  } else if (getVFS().exists(UpmemDir)) {
    PathToSDK = strdup((InstalledDir + "/..").c_str());
  } else {
    PathToSDK = strdup(Path);
  }
  asprintf(&result, "%s%s", PathToSDK, Path);
  return result;
}

Tool *DPURTE::buildLinker() const {
  return new tools::dpu::Linker(*this, PathToLinkScript, PathToRtLibDirectory,
                                RtLibName, PathToRtLibLTO, PathToRtLibLTOThin,
                                PathToBootstrap, McountLibName);
}

void DPURTE::addClangTargetOptions(
    const llvm::opt::ArgList &DriverArgs, llvm::opt::ArgStringList &CC1Args,
    Action::OffloadKind DeviceOffloadKind) const {
  Generic_ELF::addClangTargetOptions(DriverArgs, CC1Args, DeviceOffloadKind);
  Arg *A = DriverArgs.getLastArg(options::OPT_O_Group);
  if (!A || !A->getOption().matches(options::OPT_O0)) {
    // In -O0 we need to keep some unused section (from the linker point of
    // view) that will be used for debug purpose
    CC1Args.push_back("-ffunction-sections");
  }
  // add stack-size-section by default to always be able to use
  // dpu_stack_analyzer
  CC1Args.push_back("-fstack-size-section");
  CC1Args.push_back("-fdata-sections");

  if (DriverArgs.hasArg(options::OPT_pg)) {
    CC1Args.push_back("-DDPU_PROFILING");
  }
}
} // namespace toolchains

namespace tools {
namespace dpu {
void Linker::ConstructJob(Compilation &C, const JobAction &JA,
                          const InputInfo &Output, const InputInfoList &Inputs,
                          const llvm::opt::ArgList &TCArgs,
                          const char *LinkingOutput) const {
  std::string Linker = getToolChain().GetProgramPath(getShortName());
  // Put additional linker options
  ArgStringList CmdArgs;
  CmdArgs.push_back("--discard-locals");

  AddLinkerInputs(getToolChain(), Inputs, TCArgs, CmdArgs, JA);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  CmdArgs.push_back("-gc-sections");
  // Must force common allocation, so that symbols with SHN_COMMON (aka .common)
  // have space allocated in WRAM. Otherwise, the linker places symbols at
  // the very beginning of memory with no allocation.
  CmdArgs.push_back("--define-common");
  if (!TCArgs.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    CmdArgs.push_back("-L");
    CmdArgs.push_back(RtLibraryPath);

    if (TCArgs.hasArg(options::OPT_flto_EQ)) {
      // Need to inject the RTE BC library into the whole chain.
      CmdArgs.push_back(llvm::StringSwitch<const char *>(
                            TCArgs.getLastArg(options::OPT_flto_EQ)->getValue())
                            .Case("thin", RtLTOThinLibrary)
                            .Default(RtLTOLibrary));
    } else if (TCArgs.hasArg(options::OPT_flto)) {
      CmdArgs.push_back(RtLTOLibrary);
    } else {
      CmdArgs.push_back("-l");
      CmdArgs.push_back(RtLibraryName);
    }

    if (TCArgs.hasArg(options::OPT_pg)) {
      CmdArgs.push_back("-l");
      CmdArgs.push_back(McountLibraryName);
    }
  }

  bool HasArgScript = false;
  for (unsigned int EachArg = 0; EachArg < CmdArgs.size(); EachArg++) {
    if ((CmdArgs[EachArg][0] == '-') &&
        (!strncmp("-T", CmdArgs[EachArg], 2) ||
         !strncmp("--script", CmdArgs[EachArg], 8))) {
      HasArgScript = true;
      break;
    }
  }

  if (!HasArgScript) {
    CmdArgs.push_back("-T");
    CmdArgs.push_back(LinkScript);

#define STR_BUFFER_SIZE 128
#define NR_TASKLETS_FMT "NR_TASKLETS=%u"
#define DEFAULT_STACK_SIZE_FMT "STACK_SIZE_DEFAULT=%u"
#define STACK_SIZE_TASKLET_X_FMT "STACK_SIZE_TASKLET_%u=%u"
#define DEFSYM_FMT "--defsym="
    uint32_t nr_running_tasklets = 1;
    uint32_t default_stack_size = 1024;
    uint32_t stack_size[DPU_NR_THREADS];
    bool nr_running_tasklets_defined = false;
    bool stack_size_defined[DPU_NR_THREADS];
    memset(stack_size, 0xff, DPU_NR_THREADS * sizeof(uint32_t));
    memset(stack_size_defined, 0x0, DPU_NR_THREADS * sizeof(bool));
    for (unsigned int EachArg = 0; EachArg < TCArgs.size(); EachArg++) {
      std::string arg(TCArgs.getArgString(EachArg));
      if (arg.find("-D") == 0) {
        const char *symbol_and_value = &(arg.c_str())[2];
        std::string next_arg;
        if (*symbol_and_value == '\0') {
          next_arg = TCArgs.getArgString(++EachArg);
          symbol_and_value = next_arg.c_str();
        }

        uint32_t tasklet_id, tasklet_stack_size;
        if (sscanf(symbol_and_value, NR_TASKLETS_FMT, &nr_running_tasklets) ==
            1) {
          continue;
        } else if (sscanf(symbol_and_value, DEFAULT_STACK_SIZE_FMT,
                          &default_stack_size) == 1) {
          continue;
        } else if (sscanf(symbol_and_value, STACK_SIZE_TASKLET_X_FMT,
                          &tasklet_id, &tasklet_stack_size) == 2) {
          stack_size[tasklet_id] = tasklet_stack_size;
        }
      } else if (arg.find("-Wl,") == 0) {
        const char *symbol_and_value = &(arg.c_str())[4];
        uint32_t tasklet_id, tasklet_stack_size;
        if (sscanf(symbol_and_value, DEFSYM_FMT NR_TASKLETS_FMT,
                   &nr_running_tasklets) == 1) {
          nr_running_tasklets_defined = true;
        } else if (sscanf(symbol_and_value, STACK_SIZE_TASKLET_X_FMT,
                          &tasklet_id, &tasklet_stack_size) == 2) {
          stack_size_defined[tasklet_id] = true;
        }
      }
    }

    if (!nr_running_tasklets_defined) {
      static char str_nr_running_tasklets[STR_BUFFER_SIZE];
      sprintf(str_nr_running_tasklets, DEFSYM_FMT NR_TASKLETS_FMT,
              nr_running_tasklets);
      CmdArgs.push_back(str_nr_running_tasklets);
    }

    static char str_tasklet_size[DPU_NR_THREADS][STR_BUFFER_SIZE];
    for (unsigned int each_tasklet = 0; each_tasklet < DPU_NR_THREADS;
         each_tasklet++) {
      if (stack_size_defined[each_tasklet]) {
        continue;
      } else if (each_tasklet < nr_running_tasklets &&
                 stack_size[each_tasklet] == 0xffffffff) {
        sprintf(str_tasklet_size[each_tasklet],
                DEFSYM_FMT STACK_SIZE_TASKLET_X_FMT, each_tasklet,
                default_stack_size);
      } else if (stack_size[each_tasklet] == 0xffffffff) {
        sprintf(str_tasklet_size[each_tasklet],
                DEFSYM_FMT STACK_SIZE_TASKLET_X_FMT, each_tasklet, 0);
      } else {
        sprintf(str_tasklet_size[each_tasklet],
                DEFSYM_FMT STACK_SIZE_TASKLET_X_FMT, each_tasklet,
                stack_size[each_tasklet]);
      }
      CmdArgs.push_back(str_tasklet_size[each_tasklet]);
    }
  }

  if (!TCArgs.hasArg(options::OPT_nostartfiles)) {
    CmdArgs.push_back(Bootstrap);
  }

  /* Pass -L options to the linker */
  TCArgs.AddAllArgs(CmdArgs, options::OPT_L);

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), TCArgs.MakeArgString(Linker), CmdArgs, Inputs));
}
} // namespace dpu
} // namespace tools
} // namespace driver
} // namespace clang
