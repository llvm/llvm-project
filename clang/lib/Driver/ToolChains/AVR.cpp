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
#include "llvm/ADT/StringExtras.h"
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

const struct {
  StringRef Name;
  std::string SubPath;
  StringRef Family;
} MCUInfo[] = {
    {"at90s1200", "", "avr1"},
    {"attiny11", "", "avr1"},
    {"attiny12", "", "avr1"},
    {"attiny15", "", "avr1"},
    {"attiny28", "", "avr1"},
    {"at90s2313", "tiny-stack", "avr2"},
    {"at90s2323", "tiny-stack", "avr2"},
    {"at90s2333", "tiny-stack", "avr2"},
    {"at90s2343", "tiny-stack", "avr2"},
    {"at90s4433", "tiny-stack", "avr2"},
    {"attiny22", "tiny-stack", "avr2"},
    {"attiny26", "tiny-stack", "avr2"},
    {"at90s4414", "", "avr2"},
    {"at90s4434", "", "avr2"},
    {"at90s8515", "", "avr2"},
    {"at90c8534", "", "avr2"},
    {"at90s8535", "", "avr2"},
    {"attiny13", "avr25/tiny-stack", "avr25"},
    {"attiny13a", "avr25/tiny-stack", "avr25"},
    {"attiny2313", "avr25/tiny-stack", "avr25"},
    {"attiny2313a", "avr25/tiny-stack", "avr25"},
    {"attiny24", "avr25/tiny-stack", "avr25"},
    {"attiny24a", "avr25/tiny-stack", "avr25"},
    {"attiny25", "avr25/tiny-stack", "avr25"},
    {"attiny261", "avr25/tiny-stack", "avr25"},
    {"attiny261a", "avr25/tiny-stack", "avr25"},
    {"at86rf401", "avr25", "avr25"},
    {"ata5272", "avr25", "avr25"},
    {"attiny4313", "avr25", "avr25"},
    {"attiny44", "avr25", "avr25"},
    {"attiny44a", "avr25", "avr25"},
    {"attiny84", "avr25", "avr25"},
    {"attiny84a", "avr25", "avr25"},
    {"attiny45", "avr25", "avr25"},
    {"attiny85", "avr25", "avr25"},
    {"attiny441", "avr25", "avr25"},
    {"attiny461", "avr25", "avr25"},
    {"attiny461a", "avr25", "avr25"},
    {"attiny841", "avr25", "avr25"},
    {"attiny861", "avr25", "avr25"},
    {"attiny861a", "avr25", "avr25"},
    {"attiny87", "avr25", "avr25"},
    {"attiny43u", "avr25", "avr25"},
    {"attiny48", "avr25", "avr25"},
    {"attiny88", "avr25", "avr25"},
    {"attiny828", "avr25", "avr25"},
    {"at43usb355", "avr3", "avr3"},
    {"at76c711", "avr3", "avr3"},
    {"atmega103", "avr31", "avr31"},
    {"at43usb320", "avr31", "avr31"},
    {"attiny167", "avr35", "avr35"},
    {"at90usb82", "avr35", "avr35"},
    {"at90usb162", "avr35", "avr35"},
    {"ata5505", "avr35", "avr35"},
    {"atmega8u2", "avr35", "avr35"},
    {"atmega16u2", "avr35", "avr35"},
    {"atmega32u2", "avr35", "avr35"},
    {"attiny1634", "avr35", "avr35"},
    {"atmega8", "avr4", "avr4"},
    {"ata6289", "avr4", "avr4"},
    {"atmega8a", "avr4", "avr4"},
    {"ata6285", "avr4", "avr4"},
    {"ata6286", "avr4", "avr4"},
    {"atmega48", "avr4", "avr4"},
    {"atmega48a", "avr4", "avr4"},
    {"atmega48pa", "avr4", "avr4"},
    {"atmega48pb", "avr4", "avr4"},
    {"atmega48p", "avr4", "avr4"},
    {"atmega88", "avr4", "avr4"},
    {"atmega88a", "avr4", "avr4"},
    {"atmega88p", "avr4", "avr4"},
    {"atmega88pa", "avr4", "avr4"},
    {"atmega88pb", "avr4", "avr4"},
    {"atmega8515", "avr4", "avr4"},
    {"atmega8535", "avr4", "avr4"},
    {"atmega8hva", "avr4", "avr4"},
    {"at90pwm1", "avr4", "avr4"},
    {"at90pwm2", "avr4", "avr4"},
    {"at90pwm2b", "avr4", "avr4"},
    {"at90pwm3", "avr4", "avr4"},
    {"at90pwm3b", "avr4", "avr4"},
    {"at90pwm81", "avr4", "avr4"},
    {"ata5790", "avr5", "avr5"},
    {"ata5795", "avr5", "avr5"},
    {"atmega16", "avr5", "avr5"},
    {"atmega16a", "avr5", "avr5"},
    {"atmega161", "avr5", "avr5"},
    {"atmega162", "avr5", "avr5"},
    {"atmega163", "avr5", "avr5"},
    {"atmega164a", "avr5", "avr5"},
    {"atmega164p", "avr5", "avr5"},
    {"atmega164pa", "avr5", "avr5"},
    {"atmega165", "avr5", "avr5"},
    {"atmega165a", "avr5", "avr5"},
    {"atmega165p", "avr5", "avr5"},
    {"atmega165pa", "avr5", "avr5"},
    {"atmega168", "avr5", "avr5"},
    {"atmega168a", "avr5", "avr5"},
    {"atmega168p", "avr5", "avr5"},
    {"atmega168pa", "avr5", "avr5"},
    {"atmega168pb", "avr5", "avr5"},
    {"atmega169", "avr5", "avr5"},
    {"atmega169a", "avr5", "avr5"},
    {"atmega169p", "avr5", "avr5"},
    {"atmega169pa", "avr5", "avr5"},
    {"atmega32", "avr5", "avr5"},
    {"atmega32a", "avr5", "avr5"},
    {"atmega323", "avr5", "avr5"},
    {"atmega324a", "avr5", "avr5"},
    {"atmega324p", "avr5", "avr5"},
    {"atmega324pa", "avr5", "avr5"},
    {"atmega325", "avr5", "avr5"},
    {"atmega325a", "avr5", "avr5"},
    {"atmega325p", "avr5", "avr5"},
    {"atmega325pa", "avr5", "avr5"},
    {"atmega3250", "avr5", "avr5"},
    {"atmega3250a", "avr5", "avr5"},
    {"atmega3250p", "avr5", "avr5"},
    {"atmega3250pa", "avr5", "avr5"},
    {"atmega328", "avr5", "avr5"},
    {"atmega328p", "avr5", "avr5"},
    {"atmega329", "avr5", "avr5"},
    {"atmega329a", "avr5", "avr5"},
    {"atmega329p", "avr5", "avr5"},
    {"atmega329pa", "avr5", "avr5"},
    {"atmega3290", "avr5", "avr5"},
    {"atmega3290a", "avr5", "avr5"},
    {"atmega3290p", "avr5", "avr5"},
    {"atmega3290pa", "avr5", "avr5"},
    {"atmega406", "avr5", "avr5"},
    {"atmega64", "avr5", "avr5"},
    {"atmega64a", "avr5", "avr5"},
    {"atmega640", "avr5", "avr5"},
    {"atmega644", "avr5", "avr5"},
    {"atmega644a", "avr5", "avr5"},
    {"atmega644p", "avr5", "avr5"},
    {"atmega644pa", "avr5", "avr5"},
    {"atmega645", "avr5", "avr5"},
    {"atmega645a", "avr5", "avr5"},
    {"atmega645p", "avr5", "avr5"},
    {"atmega649", "avr5", "avr5"},
    {"atmega649a", "avr5", "avr5"},
    {"atmega649p", "avr5", "avr5"},
    {"atmega6450", "avr5", "avr5"},
    {"atmega6450a", "avr5", "avr5"},
    {"atmega6450p", "avr5", "avr5"},
    {"atmega6490", "avr5", "avr5"},
    {"atmega6490a", "avr5", "avr5"},
    {"atmega6490p", "avr5", "avr5"},
    {"atmega64rfr2", "avr5", "avr5"},
    {"atmega644rfr2", "avr5", "avr5"},
    {"atmega16hva", "avr5", "avr5"},
    {"atmega16hva2", "avr5", "avr5"},
    {"atmega16hvb", "avr5", "avr5"},
    {"atmega16hvbrevb", "avr5", "avr5"},
    {"atmega32hvb", "avr5", "avr5"},
    {"atmega32hvbrevb", "avr5", "avr5"},
    {"atmega64hve", "avr5", "avr5"},
    {"at90can32", "avr5", "avr5"},
    {"at90can64", "avr5", "avr5"},
    {"at90pwm161", "avr5", "avr5"},
    {"at90pwm216", "avr5", "avr5"},
    {"at90pwm316", "avr5", "avr5"},
    {"atmega32c1", "avr5", "avr5"},
    {"atmega64c1", "avr5", "avr5"},
    {"atmega16m1", "avr5", "avr5"},
    {"atmega32m1", "avr5", "avr5"},
    {"atmega64m1", "avr5", "avr5"},
    {"atmega16u4", "avr5", "avr5"},
    {"atmega32u4", "avr5", "avr5"},
    {"atmega32u6", "avr5", "avr5"},
    {"at90usb646", "avr5", "avr5"},
    {"at90usb647", "avr5", "avr5"},
    {"at90scr100", "avr5", "avr5"},
    {"at94k", "avr5", "avr5"},
    {"m3000", "avr5", "avr5"},
    {"atmega128", "avr51", "avr51"},
    {"atmega128a", "avr51", "avr51"},
    {"atmega1280", "avr51", "avr51"},
    {"atmega1281", "avr51", "avr51"},
    {"atmega1284", "avr51", "avr51"},
    {"atmega1284p", "avr51", "avr51"},
    {"atmega128rfa1", "avr51", "avr51"},
    {"atmega128rfr2", "avr51", "avr51"},
    {"atmega1284rfr2", "avr51", "avr51"},
    {"at90can128", "avr51", "avr51"},
    {"at90usb1286", "avr51", "avr51"},
    {"at90usb1287", "avr51", "avr51"},
    {"atmega2560", "avr6", "avr6"},
    {"atmega2561", "avr6", "avr6"},
    {"atmega256rfr2", "avr6", "avr6"},
    {"atmega2564rfr2", "avr6", "avr6"},
    {"attiny4", "avrtiny", "avrtiny"},
    {"attiny5", "avrtiny", "avrtiny"},
    {"attiny9", "avrtiny", "avrtiny"},
    {"attiny10", "avrtiny", "avrtiny"},
    {"attiny20", "avrtiny", "avrtiny"},
    {"attiny40", "avrtiny", "avrtiny"},
    {"atxmega16a4", "avrxmega2", "avrxmega2"},
    {"atxmega16a4u", "avrxmega2", "avrxmega2"},
    {"atxmega16c4", "avrxmega2", "avrxmega2"},
    {"atxmega16d4", "avrxmega2", "avrxmega2"},
    {"atxmega32a4", "avrxmega2", "avrxmega2"},
    {"atxmega32a4u", "avrxmega2", "avrxmega2"},
    {"atxmega32c4", "avrxmega2", "avrxmega2"},
    {"atxmega32d4", "avrxmega2", "avrxmega2"},
    {"atxmega32e5", "avrxmega2", "avrxmega2"},
    {"atxmega16e5", "avrxmega2", "avrxmega2"},
    {"atxmega8e5", "avrxmega2", "avrxmega2"},
    {"atxmega64a3u", "avrxmega4", "avrxmega4"},
    {"atxmega64a4u", "avrxmega4", "avrxmega4"},
    {"atxmega64b1", "avrxmega4", "avrxmega4"},
    {"atxmega64b3", "avrxmega4", "avrxmega4"},
    {"atxmega64c3", "avrxmega4", "avrxmega4"},
    {"atxmega64d3", "avrxmega4", "avrxmega4"},
    {"atxmega64d4", "avrxmega4", "avrxmega4"},
    {"atxmega64a1", "avrxmega5", "avrxmega5"},
    {"atxmega64a1u", "avrxmega5", "avrxmega5"},
    {"atxmega128a3", "avrxmega6", "avrxmega6"},
    {"atxmega128a3u", "avrxmega6", "avrxmega6"},
    {"atxmega128b1", "avrxmega6", "avrxmega6"},
    {"atxmega128b3", "avrxmega6", "avrxmega6"},
    {"atxmega128c3", "avrxmega6", "avrxmega6"},
    {"atxmega128d3", "avrxmega6", "avrxmega6"},
    {"atxmega128d4", "avrxmega6", "avrxmega6"},
    {"atxmega192a3", "avrxmega6", "avrxmega6"},
    {"atxmega192a3u", "avrxmega6", "avrxmega6"},
    {"atxmega192c3", "avrxmega6", "avrxmega6"},
    {"atxmega192d3", "avrxmega6", "avrxmega6"},
    {"atxmega256a3", "avrxmega6", "avrxmega6"},
    {"atxmega256a3u", "avrxmega6", "avrxmega6"},
    {"atxmega256a3b", "avrxmega6", "avrxmega6"},
    {"atxmega256a3bu", "avrxmega6", "avrxmega6"},
    {"atxmega256c3", "avrxmega6", "avrxmega6"},
    {"atxmega256d3", "avrxmega6", "avrxmega6"},
    {"atxmega384c3", "avrxmega6", "avrxmega6"},
    {"atxmega384d3", "avrxmega6", "avrxmega6"},
    {"atxmega128a1", "avrxmega7", "avrxmega7"},
    {"atxmega128a1u", "avrxmega7", "avrxmega7"},
    {"atxmega128a4u", "avrxmega7", "avrxmega7"},
};

std::string GetMCUSubPath(StringRef MCUName) {
  for (const auto &MCU : MCUInfo)
    if (MCU.Name == MCUName)
      return std::string(MCU.SubPath);
  return "";
}

llvm::Optional<StringRef> GetMCUFamilyName(StringRef MCUName) {
  for (const auto &MCU : MCUInfo)
    if (MCU.Name == MCUName)
      return Optional<StringRef>(MCU.Family);
  return Optional<StringRef>();
}

llvm::Optional<unsigned> GetMCUSectionAddressData(StringRef MCU) {
  return llvm::StringSwitch<llvm::Optional<unsigned>>(MCU)
      .Case("atmega328", Optional<unsigned>(0x800100))
      .Case("atmega328p", Optional<unsigned>(0x800100))
      .Default(Optional<unsigned>());
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
      Optional<StringRef> FamilyName = GetMCUFamilyName(CPU);
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
        std::string SubPath = GetMCUSubPath(CPU);

        getFilePaths().push_back(LibcRoot + std::string("/lib/") + SubPath);
        getFilePaths().push_back(GCCRoot + std::string("/") + SubPath);

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
                               const InputInfoList &Inputs, const ArgList &Args,
                               const char *LinkingOutput) const {
  // Compute information about the target AVR.
  std::string CPU = getCPUName(Args, getToolChain().getTriple());
  llvm::Optional<StringRef> FamilyName = GetMCUFamilyName(CPU);
  llvm::Optional<unsigned> SectionAddressData = GetMCUSectionAddressData(CPU);

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

  if (SectionAddressData.hasValue()) {
    std::string DataSectionArg = std::string("-Tdata=0x") +
                                 llvm::utohexstr(SectionAddressData.getValue());
    CmdArgs.push_back(Args.MakeArgString(DataSectionArg));
  } else {
    // We do not have an entry for this CPU in the address mapping table yet.
    getToolChain().getDriver().Diag(
        diag::warn_drv_avr_linker_section_addresses_not_implemented)
        << CPU;
  }

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

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Args.MakeArgString(Linker),
      CmdArgs, Inputs, Output));
}

llvm::Optional<std::string> AVRToolChain::findAVRLibcInstallation() const {
  for (StringRef PossiblePath : PossibleAVRLibcLocations) {
    // Return the first avr-libc installation that exists.
    if (llvm::sys::fs::is_directory(PossiblePath))
      return Optional<std::string>(std::string(PossiblePath));
  }

  return llvm::None;
}
