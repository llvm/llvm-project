//===--- CommonArgs.h - Args handling for multiple toolchains ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H

#include "InputInfo.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Support/CodeGen.h"

namespace clang {
namespace driver {
namespace tools {

void addPathIfExists(const Driver &D, const Twine &Path,
                     ToolChain::path_list &Paths);

void AddLinkerInputs(const ToolChain &TC, const InputInfoList &Inputs,
                     const llvm::opt::ArgList &Args,
                     llvm::opt::ArgStringList &CmdArgs, const JobAction &JA);

void claimNoWarnArgs(const llvm::opt::ArgList &Args);

bool addSanitizerRuntimes(const ToolChain &TC, const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs);

void linkSanitizerRuntimeDeps(const ToolChain &TC,
                              llvm::opt::ArgStringList &CmdArgs);

bool addXRayRuntime(const ToolChain &TC, const llvm::opt::ArgList &Args,
                    llvm::opt::ArgStringList &CmdArgs);

void linkXRayRuntimeDeps(const ToolChain &TC,
                         llvm::opt::ArgStringList &CmdArgs);

void AddRunTimeLibs(const ToolChain &TC, const Driver &D,
                    llvm::opt::ArgStringList &CmdArgs,
                    const llvm::opt::ArgList &Args);

void AddHIPLinkerScript(const ToolChain &TC, Compilation &C,
                        const InputInfo &Output, const InputInfoList &Inputs,
                        const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs, const JobAction &JA,
                        const Tool &T);

const char *SplitDebugName(const llvm::opt::ArgList &Args,
                           const InputInfo &Input, const InputInfo &Output);

void SplitDebugInfo(const ToolChain &TC, Compilation &C, const Tool &T,
                    const JobAction &JA, const llvm::opt::ArgList &Args,
                    const InputInfo &Output, const char *OutFile);

void AddGoldPlugin(const ToolChain &ToolChain, const llvm::opt::ArgList &Args,
                   llvm::opt::ArgStringList &CmdArgs, const InputInfo &Output,
                   const InputInfo &Input, bool IsThinLTO);

std::tuple<llvm::Reloc::Model, unsigned, bool>
ParsePICArgs(const ToolChain &ToolChain, const llvm::opt::ArgList &Args);

unsigned ParseFunctionAlignment(const ToolChain &TC,
                                const llvm::opt::ArgList &Args);

unsigned ParseDebugDefaultVersion(const ToolChain &TC,
                                  const llvm::opt::ArgList &Args);

void AddAssemblerKPIC(const ToolChain &ToolChain,
                      const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs);

void addArchSpecificRPath(const ToolChain &TC, const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs);
/// Returns true, if an OpenMP runtime has been added.
bool addOpenMPRuntime(llvm::opt::ArgStringList &CmdArgs, const ToolChain &TC,
                      const llvm::opt::ArgList &Args,
                      bool ForceStaticHostRuntime = false,
                      bool IsOffloadingHost = false, bool GompNeedsRT = false);

llvm::opt::Arg *getLastProfileUseArg(const llvm::opt::ArgList &Args);
llvm::opt::Arg *getLastProfileSampleUseArg(const llvm::opt::ArgList &Args);

bool isObjCAutoRefCount(const llvm::opt::ArgList &Args);

unsigned getLTOParallelism(const llvm::opt::ArgList &Args, const Driver &D);

bool areOptimizationsEnabled(const llvm::opt::ArgList &Args);

bool isUseSeparateSections(const llvm::Triple &Triple);

void addDirectoryList(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs, const char *ArgName,
                      const char *EnvVar);

void AddTargetFeature(const llvm::opt::ArgList &Args,
                      std::vector<StringRef> &Features,
                      llvm::opt::OptSpecifier OnOpt,
                      llvm::opt::OptSpecifier OffOpt, StringRef FeatureName);

std::string getCPUName(const llvm::opt::ArgList &Args, const llvm::Triple &T,
                       bool FromAs = false);

void handleTargetFeaturesGroup(const llvm::opt::ArgList &Args,
                               std::vector<StringRef> &Features,
                               llvm::opt::OptSpecifier Group);

/// Handles the -save-stats option and returns the filename to save statistics
/// to.
SmallString<128> getStatsFileName(const llvm::opt::ArgList &Args,
                                  const InputInfo &Output,
                                  const InputInfo &Input, const Driver &D);

/// \p Flag must be a flag accepted by the driver with its leading '-' removed,
//     otherwise '-print-multi-lib' will not emit them correctly.
void addMultilibFlag(bool Enabled, const char *const Flag,
                     Multilib::flags_list &Flags);

} // end namespace tools
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H
