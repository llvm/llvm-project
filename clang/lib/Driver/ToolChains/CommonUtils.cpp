//===--- CommonUtils.h - Common utilities for the toolchains ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CommonUtils.h"
#include "clang/Driver/Driver.h"
#include "llvm/ADT/SmallString.h"

using namespace clang::driver;

namespace clang {

void EscapeSpacesAndBackslashes(const char *Arg,
                                llvm::SmallVectorImpl<char> &Res) {
  for (; *Arg; ++Arg) {
    switch (*Arg) {
    default:
      break;
    case ' ':
    case '\\':
      Res.push_back('\\');
      break;
    }
    Res.push_back(*Arg);
  }
}

const char *RenderEscapedCommandLine(const ToolChain &TC,
                                     const llvm::opt::ArgList &Args) {
  const Driver &D = TC.getDriver();
  const char *Exec = D.getClangProgramPath();

  llvm::opt::ArgStringList OriginalArgs;
  for (const auto &Arg : Args)
    Arg->render(Args, OriginalArgs);

  llvm::SmallString<256> Flags;
  EscapeSpacesAndBackslashes(Exec, Flags);
  for (const char *OriginalArg : OriginalArgs) {
    llvm::SmallString<128> EscapedArg;
    EscapeSpacesAndBackslashes(OriginalArg, EscapedArg);
    Flags += " ";
    Flags += EscapedArg;
  }

  return Args.MakeArgString(Flags);
}

bool ShouldRecordCommandLine(const ToolChain &TC,
                             const llvm::opt::ArgList &Args,
                             bool &FRecordCommandLine,
                             bool &GRecordCommandLine) {
  const Driver &D = TC.getDriver();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  FRecordCommandLine =
      Args.hasFlag(options::OPT_frecord_command_line,
                   options::OPT_fno_record_command_line, false);
  GRecordCommandLine =
      Args.hasFlag(options::OPT_grecord_command_line,
                   options::OPT_gno_record_command_line, false);
  if (FRecordCommandLine && !Triple.isOSBinFormatELF() &&
      !Triple.isOSBinFormatXCOFF() && !Triple.isOSBinFormatMachO())
    D.Diag(diag::err_drv_unsupported_opt_for_target)
        << Args.getLastArg(options::OPT_frecord_command_line)->getAsString(Args)
        << TripleStr;

  return FRecordCommandLine || TC.UseDwarfDebugFlags() || GRecordCommandLine;
}

} // namespace clang
