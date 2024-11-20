//===--- UEFI.cpp - UEFI ToolChain Implementations -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UEFI.h"
#include "CommonArgs.h"
#include "Darwin.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Host.h"

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

UEFI::UEFI(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : ToolChain(D, Triple, Args) {}

Tool *UEFI::buildLinker() const { return new tools::uefi::Linker(*this); }

void tools::uefi::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  auto &TC = static_cast<const toolchains::UEFI &>(getToolChain());

  assert((Output.isFilename() || Output.isNothing()) && "invalid output");
  if (Output.isFilename())
    CmdArgs.push_back(
        Args.MakeArgString(std::string("-out:") + Output.getFilename()));

  CmdArgs.push_back("-nologo");

  // TODO: Other UEFI binary subsystems that are currently unsupported:
  // efi_boot_service_driver, efi_rom, efi_runtime_driver.
  CmdArgs.push_back("-subsystem:efi_application");

  // Default entry function name according to the TianoCore reference
  // implementation is EfiMain.
  // TODO: Provide a flag to override the entry function name.
  CmdArgs.push_back("-entry:EfiMain");

  // "Terminal Service Aware" flag is not needed for UEFI applications.
  CmdArgs.push_back("-tsaware:no");

  // EFI_APPLICATION to be linked as DLL by default.
  CmdArgs.push_back("-dll");

  if (Args.hasArg(options::OPT_g_Group, options::OPT__SLASH_Z7))
    CmdArgs.push_back("-debug");

  Args.AddAllArgValues(CmdArgs, options::OPT__SLASH_link);

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  // This should ideally be handled by ToolChain::GetLinkerPath but we need
  // to special case some linker paths. In the case of lld, we need to
  // translate 'lld' into 'lld-link'.
  StringRef Linker =
      Args.getLastArgValue(options::OPT_fuse_ld_EQ, CLANG_DEFAULT_LINKER);
  if (Linker.empty() || Linker == "lld")
    Linker = "lld-link";

  auto LinkerPath = TC.GetProgramPath(Linker.str().c_str());
  auto LinkCmd = std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileUTF16(),
      Args.MakeArgString(LinkerPath), CmdArgs, Inputs, Output);
  C.addCommand(std::move(LinkCmd));
}
