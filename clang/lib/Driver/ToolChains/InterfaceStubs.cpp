//===---  InterfaceStubs.cpp - Base InterfaceStubs Implementations C++  ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InterfaceStubs.h"
#include "CommonArgs.h"
#include "clang/Driver/Compilation.h"

namespace clang {
namespace driver {
namespace tools {
namespace ifstool {
void Merger::ConstructJob(Compilation &C, const JobAction &JA,
                          const InputInfo &Output, const InputInfoList &Inputs,
                          const llvm::opt::ArgList &Args,
                          const char *LinkingOutput) const {
  std::string Merger = getToolChain().GetProgramPath(getShortName());
  llvm::opt::ArgStringList CmdArgs;
  CmdArgs.push_back("-action");
  CmdArgs.push_back(Args.getLastArg(options::OPT_emit_merged_ifs)
                        ? "write-ifs"
                        : "write-bin");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  for (const auto &Input : Inputs)
    CmdArgs.push_back(Input.getFilename());
  C.addCommand(std::make_unique<Command>(JA, *this, Args.MakeArgString(Merger),
                                         CmdArgs, Inputs));
}
} // namespace ifstool
} // namespace tools
} // namespace driver
} // namespace clang
