//===--- DPURTE.h - DPU RTE ToolChain Implementations -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_DPURTE_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_DPURTE_H

#include "Gnu.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Option/ArgList.h"
#include <stdlib.h>

namespace clang {
namespace driver {
namespace toolchains {

namespace dpu {
void addDPUTargetOptions(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs);
} // namespace dpu

class LLVM_LIBRARY_VISIBILITY DPURTE : public Generic_ELF {
public:
  DPURTE(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args)
      : Generic_ELF(D, Triple, Args) {
    PathToStdlibIncludes = GetUpmemSdkPath("/share/upmem/include/stdlib");
    PathToSyslibIncludes = GetUpmemSdkPath("/share/upmem/include/syslib");
    PathToLinkScript = GetUpmemSdkPath("/share/upmem/include/link/dpu.lds");
    PathToRtLibDirectory = GetUpmemSdkPath("/share/upmem/include/built-in");
    RtLibName = Args.hasArg(options::OPT_pg) ? "rt_p" : "rt";
    if (Args.hasArg(options::OPT_pg)) {
      PathToRtLibLTO =
          GetUpmemSdkPath("/share/upmem/include/built-in/librtlto_p.a");
      PathToRtLibLTOThin =
          GetUpmemSdkPath("/share/upmem/include/built-in/librtltothin_p.a");
      PathToBootstrap = GetUpmemSdkPath("/share/upmem/include/misc/crt0_p.o");
    } else {
      PathToRtLibLTO =
          GetUpmemSdkPath("/share/upmem/include/built-in/librtlto.a");
      PathToRtLibLTOThin =
          GetUpmemSdkPath("/share/upmem/include/built-in/librtltothin.a");
      PathToBootstrap = GetUpmemSdkPath("/share/upmem/include/misc/crt0.o");
    }
    McountLibName = "rtmcount";
  }

  ~DPURTE() override {
    free(PathToStdlibIncludes);
    free(PathToSyslibIncludes);
    free(PathToLinkScript);
    free(PathToRtLibDirectory);
    free(PathToRtLibLTO);
    free(PathToRtLibLTOThin);
  }

  SanitizerMask getSupportedSanitizers() const override {
    SanitizerMask Res = ToolChain::getSupportedSanitizers();
    // Safe stack not supported yet Res |= SanitizerKind::SafeStack;
    return Res;
  }

  bool IsIntegratedAssemblerDefault() const override { return true; }

  bool HasNativeLLVMSupport() const override { return true; }

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  void
  addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        Action::OffloadKind DeviceOffloadKind) const override;

protected:
  Tool *buildLinker() const override;

private:
  char *GetUpmemSdkPath(const char *Path);

  char *PathToSDK = NULL;
  char *PathToSyslibIncludes;
  char *PathToStdlibIncludes;
  char *PathToLinkScript;
  char *PathToRtLibDirectory;
  const char *RtLibName;
  char *PathToRtLibLTO;
  char *PathToRtLibLTOThin;
  char *PathToBootstrap;
  const char *McountLibName;
};
} // end namespace toolchains
namespace tools {
namespace dpu {
class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC, const char *Script, const char *RtLibDir,
         const char *RtLibName, const char *PathToRtLibLTO,
         const char *PathToRtLibLTOThin, const char *PathToBootstrap,
         const char *McountLibName)
      : Tool("dpu::Linker", "ld.lld", TC) {
    LinkScript = Script;
    RtLibraryPath = RtLibDir;
    RtLibraryName = RtLibName;
    RtLTOLibrary = PathToRtLibLTO;
    RtLTOThinLibrary = PathToRtLibLTOThin;
    Bootstrap = PathToBootstrap;
    McountLibraryName = McountLibName;
  }

  bool isLinkJob() const override { return true; }

  bool hasIntegratedCPP() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  const char *LinkScript;
  const char *RtLibraryPath;
  const char *RtLibraryName;
  const char *RtLTOLibrary;
  const char *RtLTOThinLibrary;
  const char *Bootstrap;
  const char *McountLibraryName;
};
} // end namespace dpu
} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_DPURTE_H
