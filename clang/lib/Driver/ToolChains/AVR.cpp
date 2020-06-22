//===--- AVR.cpp - AVR ToolChain Implementations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AVR.h"
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

namespace {

// TODO: Consider merging this into the AVR device table
// array in Targets/AVR.cpp.
llvm::Optional<StringRef> GetMcuFamilyName(StringRef MCU) {
  return llvm::StringSwitch<llvm::Optional<StringRef>>(MCU)
      .Case("atmega328", Optional<StringRef>("avr5"))
      .Case("atmega328p", Optional<StringRef>("avr5"))
      .Default(Optional<StringRef>());
}

const StringRef PossibleAVRLibcLocations[] = {
    "/usr/avr",
    "/usr/lib/avr",
};

} // end anonymous namespace

/// AVR Toolchain
AVRToolChain::AVRToolChain(const Driver &D, const llvm::Triple &Triple,
                           const ArgList &Args)
    : Generic_ELF(D, Triple, Args), LinkStdlib(false) {
  GCCInstallation.init(Triple, Args);

  // Only add default libraries if the user hasn't explicitly opted out.
  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs) &&
      !Args.hasArg(options::OPT_c /* does not apply when not linking */)) {
    std::string CPU = getCPUName(Args, Triple);

    if (CPU.empty()) {
      // We cannot link any standard libraries without an MCU specified.
      D.Diag(diag::warn_drv_avr_mcu_not_specified);
    } else {
      Optional<StringRef> FamilyName = GetMcuFamilyName(CPU);
      Optional<std::string> AVRLibcRoot = findAVRLibcInstallation();

      if (!FamilyName.hasValue()) {
        // We do not have an entry for this CPU in the family
        // mapping table yet.
        D.Diag(diag::warn_drv_avr_family_linking_stdlibs_not_implemented)
            << CPU;
      } else if (!GCCInstallation.isValid()) {
        // No avr-gcc found and so no runtime linked.
        D.Diag(diag::warn_drv_avr_gcc_not_found);
      } else if (!AVRLibcRoot.hasValue()) {
        // No avr-libc found and so no runtime linked.
        D.Diag(diag::warn_drv_avr_libc_not_found);
      } else { // We have enough information to link stdlibs
        std::string GCCRoot = std::string(GCCInstallation.getInstallPath());
        std::string LibcRoot = AVRLibcRoot.getValue();

        getFilePaths().push_back(LibcRoot + std::string("/lib/") +
                                 std::string(*FamilyName));
        getFilePaths().push_back(GCCRoot + std::string("/") +
                                 std::string(*FamilyName));

        LinkStdlib = true;
      }
    }

    if (!LinkStdlib)
      D.Diag(diag::warn_drv_avr_stdlib_not_linked);
  }
}

Tool *AVRToolChain::buildLinker() const {
  return new tools::AVR::Linker(getTriple(), *this, LinkStdlib);
}

void AVR::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                               const InputInfo &Output,
                               const InputInfoList &Inputs,
                               const ArgList &Args,
                               const char *LinkingOutput) const {
  // Compute information about the target AVR.
  std::string CPU = getCPUName(Args, getToolChain().getTriple());
  llvm::Optional<StringRef> FamilyName = GetMcuFamilyName(CPU);

  std::string Linker = getToolChain().GetProgramPath(getShortName());
  ArgStringList CmdArgs;
  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  // Enable garbage collection of unused sections.
  CmdArgs.push_back("--gc-sections");

  // Add library search paths before we specify libraries.
  Args.AddAllArgs(CmdArgs, options::OPT_L);
  getToolChain().AddFilePathLibArgs(Args, CmdArgs);

  //   "Not [sic] that addr must be offset by adding 0x800000 the to
  //    real SRAM address so that the linker knows that the address
  //    is in the SRAM memory space."
  //
  //      - https://www.nongnu.org/avr-libc/user-manual/mem_sections.html
  CmdArgs.push_back("-Tdata=0x800100");

  // If the family name is known, we can link with the device-specific libgcc.
  // Without it, libgcc will simply not be linked. This matches avr-gcc
  // behavior.
  if (LinkStdlib) {
    assert(!CPU.empty() && "CPU name must be known in order to link stdlibs");

    // Add the object file for the CRT.
    std::string CrtFileName = std::string("-l:crt") + CPU + std::string(".o");
    CmdArgs.push_back(Args.MakeArgString(CrtFileName));

    CmdArgs.push_back("-lgcc");
    CmdArgs.push_back("-lm");
    CmdArgs.push_back("-lc");

    // Add the link library specific to the MCU.
    CmdArgs.push_back(Args.MakeArgString(std::string("-l") + CPU));

    // Specify the family name as the emulation mode to use.
    // This is almost always required because otherwise avr-ld
    // will assume 'avr2' and warn about the program being larger
    // than the bare minimum supports.
    CmdArgs.push_back(Args.MakeArgString(std::string("-m") + *FamilyName));
  }

  C.addCommand(std::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                          CmdArgs, Inputs));
}

llvm::Optional<std::string> AVRToolChain::findAVRLibcInstallation() const {
  for (StringRef PossiblePath : PossibleAVRLibcLocations) {
    // Return the first avr-libc installation that exists.
    if (llvm::sys::fs::is_directory(PossiblePath))
      return Optional<std::string>(std::string(PossiblePath));
  }

  return llvm::None;
}
