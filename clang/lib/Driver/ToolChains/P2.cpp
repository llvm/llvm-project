//===--- P2.cpp - P2 ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <iostream>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;


const StringRef PossibleP2LibCLocations[] = {
    "/opt/p2llvm/libc",
};

const StringRef PossibleP2LibP2Locations[] = {
    "/opt/p2llvm/libp2",
};

/// P2 Toolchain
P2ToolChain::P2ToolChain(const Driver &D, const llvm::Triple &Triple,
                        const ArgList &Args) : Generic_ELF(D, Triple, Args) {

    std::string libc_dir;
    std::string libp2_dir;

    for (StringRef PossiblePath : PossibleP2LibCLocations) {
    // Return the first p2 libc installation that exists.
        if (llvm::sys::fs::is_directory(PossiblePath))
            libc_dir = std::string(PossiblePath);
    }

    for (StringRef PossiblePath : PossibleP2LibP2Locations) {
    // Return the first p2 libc installation that exists.
        if (llvm::sys::fs::is_directory(PossiblePath))
            libp2_dir = std::string(PossiblePath);
    }

    getFilePaths().push_back(libc_dir + std::string("/lib/"));
    getFilePaths().push_back(libp2_dir + std::string("/lib/"));
}

void P2ToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) const {
    if (DriverArgs.hasArg(options::OPT_nostdinc) ||
        DriverArgs.hasArg(options::OPT_nostdlibinc)) return;

    std::string sys_root = computeSysRoot();

    addSystemInclude(DriverArgs, CC1Args, sys_root + std::string("/libp2/include"));
    addSystemInclude(DriverArgs, CC1Args, sys_root + std::string("/libc/include"));
}

void P2ToolChain::addClangTargetOptions(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args,
                                            Action::OffloadKind) const {
    CC1Args.push_back("-fno-rtti");
    CC1Args.push_back("-fno-jump-tables");
}

std::string P2ToolChain::computeSysRoot() const {
    if (!getDriver().SysRoot.empty())
        return getDriver().SysRoot;

    SmallString<128> Dir;
    if (GCCInstallation.isValid())
        llvm::sys::path::append(Dir, GCCInstallation.getParentLibPath(), "..");
    else
        llvm::sys::path::append(Dir, getDriver().Dir, "..");

    return std::string(Dir.str());
}

Tool *P2ToolChain::buildLinker() const {
  return new tools::P2::Linker(getTriple(), *this);
}

void P2::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ArgList &Args,
                               const char *LinkingOutput) const {

    std::string Linker = getToolChain().GetProgramPath(getShortName());
    ArgStringList CmdArgs;
    AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

    CmdArgs.push_back("-v");

    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());

    Args.AddAllArgs(CmdArgs, options::OPT_L);
    getToolChain().AddFilePathLibArgs(Args, CmdArgs);

    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lp2");
    CmdArgs.push_back("-Tp2.ld");

    std::string sys_root = getToolChain().computeSysRoot();

    CmdArgs.push_back(Args.MakeArgString("-L" + sys_root));

    C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::AtFileCurCP(),
                                            Args.MakeArgString(Linker), CmdArgs, Inputs));
}

