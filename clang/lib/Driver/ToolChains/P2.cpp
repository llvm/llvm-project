//===--- P2.cpp - P2 ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "CommonArgs.h"
#include "InputInfo.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

/// P2 Toolchain
P2ToolChain::P2ToolChain(const Driver &D, const llvm::Triple &Triple,
                           const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  //GCCInstallation.init(Triple, Args);
}

Tool *P2ToolChain::buildLinker() const {
  return new tools::P2::Linker(getTriple(), *this);
}

void P2::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ArgList &Args,
                               const char *LinkingOutput) const {

    // std::string Linker = getToolChain().GetProgramPath(getShortName());
    // ArgStringList CmdArgs;
    // CmdArgs.push_back("-o");
    // CmdArgs.push_back(Output.getFilename());

    // C.addCommand(std::make_unique<Command>(JA, *this, Args.MakeArgString(Linker), CmdArgs, Inputs));
}

// llvm::Optional<std::string> P2ToolChain::findP2LibcInstallation() const {
//   return llvm::None;
// }
