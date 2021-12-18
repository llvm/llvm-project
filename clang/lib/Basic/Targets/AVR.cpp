//===--- AVR.cpp - Implement AVR target feature support -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements AVR TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "AVR.h"
#include "clang/Basic/MacroBuilder.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

namespace clang {
namespace targets {

/// Information about a specific microcontroller.
struct LLVM_LIBRARY_VISIBILITY MCUInfo {
  const char *Name;
  const char *DefineName;
  const int NumFlashBanks; // -1 means the device does not support LPM/ELPM.
};

// This list should be kept up-to-date with AVRDevices.td in LLVM.
static MCUInfo AVRMcus[] = {
    {"at90s1200", "__AVR_AT90S1200__", 0},
    {"attiny11", "__AVR_ATtiny11__", 0},
    {"attiny12", "__AVR_ATtiny12__", 0},
    {"attiny15", "__AVR_ATtiny15__", 0},
    {"attiny28", "__AVR_ATtiny28__", 0},
    {"at90s2313", "__AVR_AT90S2313__", 1},
    {"at90s2323", "__AVR_AT90S2323__", 1},
    {"at90s2333", "__AVR_AT90S2333__", 1},
    {"at90s2343", "__AVR_AT90S2343__", 1},
    {"attiny22", "__AVR_ATtiny22__", 1},
    {"attiny26", "__AVR_ATtiny26__", 1},
    {"at86rf401", "__AVR_AT86RF401__", 1},
    {"at90s4414", "__AVR_AT90S4414__", 1},
    {"at90s4433", "__AVR_AT90S4433__", 1},
    {"at90s4434", "__AVR_AT90S4434__", 1},
    {"at90s8515", "__AVR_AT90S8515__", 1},
    {"at90c8534", "__AVR_AT90c8534__", 1},
    {"at90s8535", "__AVR_AT90S8535__", 1},
    {"ata5272", "__AVR_ATA5272__", 1},
    {"attiny13", "__AVR_ATtiny13__", 1},
    {"attiny13a", "__AVR_ATtiny13A__", 1},
    {"attiny2313", "__AVR_ATtiny2313__", 1},
    {"attiny2313a", "__AVR_ATtiny2313A__", 1},
    {"attiny24", "__AVR_ATtiny24__", 1},
    {"attiny24a", "__AVR_ATtiny24A__", 1},
    {"attiny4313", "__AVR_ATtiny4313__", 1},
    {"attiny44", "__AVR_ATtiny44__", 1},
    {"attiny44a", "__AVR_ATtiny44A__", 1},
    {"attiny84", "__AVR_ATtiny84__", 1},
    {"attiny84a", "__AVR_ATtiny84A__", 1},
    {"attiny25", "__AVR_ATtiny25__", 1},
    {"attiny45", "__AVR_ATtiny45__", 1},
    {"attiny85", "__AVR_ATtiny85__", 1},
    {"attiny261", "__AVR_ATtiny261__", 1},
    {"attiny261a", "__AVR_ATtiny261A__", 1},
    {"attiny441", "__AVR_ATtiny441__", 1},
    {"attiny461", "__AVR_ATtiny461__", 1},
    {"attiny461a", "__AVR_ATtiny461A__", 1},
    {"attiny841", "__AVR_ATtiny841__", 1},
    {"attiny861", "__AVR_ATtiny861__", 1},
    {"attiny861a", "__AVR_ATtiny861A__", 1},
    {"attiny87", "__AVR_ATtiny87__", 1},
    {"attiny43u", "__AVR_ATtiny43U__", 1},
    {"attiny48", "__AVR_ATtiny48__", 1},
    {"attiny88", "__AVR_ATtiny88__", 1},
    {"attiny828", "__AVR_ATtiny828__", 1},
    {"at43usb355", "__AVR_AT43USB355__", 1},
    {"at76c711", "__AVR_AT76C711__", 1},
    {"atmega103", "__AVR_ATmega103__", 1},
    {"at43usb320", "__AVR_AT43USB320__", 1},
    {"attiny167", "__AVR_ATtiny167__", 1},
    {"at90usb82", "__AVR_AT90USB82__", 1},
    {"at90usb162", "__AVR_AT90USB162__", 1},
    {"ata5505", "__AVR_ATA5505__", 1},
    {"atmega8u2", "__AVR_ATmega8U2__", 1},
    {"atmega16u2", "__AVR_ATmega16U2__", 1},
    {"atmega32u2", "__AVR_ATmega32U2__", 1},
    {"attiny1634", "__AVR_ATtiny1634__", 1},
    {"atmega8", "__AVR_ATmega8__", 1},
    {"ata6289", "__AVR_ATA6289__", 1},
    {"atmega8a", "__AVR_ATmega8A__", 1},
    {"ata6285", "__AVR_ATA6285__", 1},
    {"ata6286", "__AVR_ATA6286__", 1},
    {"atmega48", "__AVR_ATmega48__", 1},
    {"atmega48a", "__AVR_ATmega48A__", 1},
    {"atmega48pa", "__AVR_ATmega48PA__", 1},
    {"atmega48pb", "__AVR_ATmega48PB__", 1},
    {"atmega48p", "__AVR_ATmega48P__", 1},
    {"atmega88", "__AVR_ATmega88__", 1},
    {"atmega88a", "__AVR_ATmega88A__", 1},
    {"atmega88p", "__AVR_ATmega88P__", 1},
    {"atmega88pa", "__AVR_ATmega88PA__", 1},
    {"atmega88pb", "__AVR_ATmega88PB__", 1},
    {"atmega8515", "__AVR_ATmega8515__", 1},
    {"atmega8535", "__AVR_ATmega8535__", 1},
    {"atmega8hva", "__AVR_ATmega8HVA__", 1},
    {"at90pwm1", "__AVR_AT90PWM1__", 1},
    {"at90pwm2", "__AVR_AT90PWM2__", 1},
    {"at90pwm2b", "__AVR_AT90PWM2B__", 1},
    {"at90pwm3", "__AVR_AT90PWM3__", 1},
    {"at90pwm3b", "__AVR_AT90PWM3B__", 1},
    {"at90pwm81", "__AVR_AT90PWM81__", 1},
    {"ata5790", "__AVR_ATA5790__", 1},
    {"ata5795", "__AVR_ATA5795__", 1},
    {"atmega16", "__AVR_ATmega16__", 1},
    {"atmega16a", "__AVR_ATmega16A__", 1},
    {"atmega161", "__AVR_ATmega161__", 1},
    {"atmega162", "__AVR_ATmega162__", 1},
    {"atmega163", "__AVR_ATmega163__", 1},
    {"atmega164a", "__AVR_ATmega164A__", 1},
    {"atmega164p", "__AVR_ATmega164P__", 1},
    {"atmega164pa", "__AVR_ATmega164PA__", 1},
    {"atmega165", "__AVR_ATmega165__", 1},
    {"atmega165a", "__AVR_ATmega165A__", 1},
    {"atmega165p", "__AVR_ATmega165P__", 1},
    {"atmega165pa", "__AVR_ATmega165PA__", 1},
    {"atmega168", "__AVR_ATmega168__", 1},
    {"atmega168a", "__AVR_ATmega168A__", 1},
    {"atmega168p", "__AVR_ATmega168P__", 1},
    {"atmega168pa", "__AVR_ATmega168PA__", 1},
    {"atmega168pb", "__AVR_ATmega168PB__", 1},
    {"atmega169", "__AVR_ATmega169__", 1},
    {"atmega169a", "__AVR_ATmega169A__", 1},
    {"atmega169p", "__AVR_ATmega169P__", 1},
    {"atmega169pa", "__AVR_ATmega169PA__", 1},
    {"atmega32", "__AVR_ATmega32__", 1},
    {"atmega32a", "__AVR_ATmega32A__", 1},
    {"atmega323", "__AVR_ATmega323__", 1},
    {"atmega324a", "__AVR_ATmega324A__", 1},
    {"atmega324p", "__AVR_ATmega324P__", 1},
    {"atmega324pa", "__AVR_ATmega324PA__", 1},
    {"atmega324pb", "__AVR_ATmega324PB__", 1},
    {"atmega325", "__AVR_ATmega325__", 1},
    {"atmega325a", "__AVR_ATmega325A__", 1},
    {"atmega325p", "__AVR_ATmega325P__", 1},
    {"atmega325pa", "__AVR_ATmega325PA__", 1},
    {"atmega3250", "__AVR_ATmega3250__", 1},
    {"atmega3250a", "__AVR_ATmega3250A__", 1},
    {"atmega3250p", "__AVR_ATmega3250P__", 1},
    {"atmega3250pa", "__AVR_ATmega3250PA__", 1},
    {"atmega328", "__AVR_ATmega328__", 1},
    {"atmega328p", "__AVR_ATmega328P__", 1},
    {"atmega328pb", "__AVR_ATmega328PB__", 1},
    {"atmega329", "__AVR_ATmega329__", 1},
    {"atmega329a", "__AVR_ATmega329A__", 1},
    {"atmega329p", "__AVR_ATmega329P__", 1},
    {"atmega329pa", "__AVR_ATmega329PA__", 1},
    {"atmega3290", "__AVR_ATmega3290__", 1},
    {"atmega3290a", "__AVR_ATmega3290A__", 1},
    {"atmega3290p", "__AVR_ATmega3290P__", 1},
    {"atmega3290pa", "__AVR_ATmega3290PA__", 1},
    {"atmega406", "__AVR_ATmega406__", 1},
    {"atmega64", "__AVR_ATmega64__", 1},
    {"atmega64a", "__AVR_ATmega64A__", 1},
    {"atmega640", "__AVR_ATmega640__", 1},
    {"atmega644", "__AVR_ATmega644__", 1},
    {"atmega644a", "__AVR_ATmega644A__", 1},
    {"atmega644p", "__AVR_ATmega644P__", 1},
    {"atmega644pa", "__AVR_ATmega644PA__", 1},
    {"atmega645", "__AVR_ATmega645__", 1},
    {"atmega645a", "__AVR_ATmega645A__", 1},
    {"atmega645p", "__AVR_ATmega645P__", 1},
    {"atmega649", "__AVR_ATmega649__", 1},
    {"atmega649a", "__AVR_ATmega649A__", 1},
    {"atmega649p", "__AVR_ATmega649P__", 1},
    {"atmega6450", "__AVR_ATmega6450__", 1},
    {"atmega6450a", "__AVR_ATmega6450A__", 1},
    {"atmega6450p", "__AVR_ATmega6450P__", 1},
    {"atmega6490", "__AVR_ATmega6490__", 1},
    {"atmega6490a", "__AVR_ATmega6490A__", 1},
    {"atmega6490p", "__AVR_ATmega6490P__", 1},
    {"atmega64rfr2", "__AVR_ATmega64RFR2__", 1},
    {"atmega644rfr2", "__AVR_ATmega644RFR2__", 1},
    {"atmega16hva", "__AVR_ATmega16HVA__", 1},
    {"atmega16hva2", "__AVR_ATmega16HVA2__", 1},
    {"atmega16hvb", "__AVR_ATmega16HVB__", 1},
    {"atmega16hvbrevb", "__AVR_ATmega16HVBREVB__", 1},
    {"atmega32hvb", "__AVR_ATmega32HVB__", 1},
    {"atmega32hvbrevb", "__AVR_ATmega32HVBREVB__", 1},
    {"atmega64hve", "__AVR_ATmega64HVE__", 1},
    {"at90can32", "__AVR_AT90CAN32__", 1},
    {"at90can64", "__AVR_AT90CAN64__", 1},
    {"at90pwm161", "__AVR_AT90PWM161__", 1},
    {"at90pwm216", "__AVR_AT90PWM216__", 1},
    {"at90pwm316", "__AVR_AT90PWM316__", 1},
    {"atmega32c1", "__AVR_ATmega32C1__", 1},
    {"atmega64c1", "__AVR_ATmega64C1__", 1},
    {"atmega16m1", "__AVR_ATmega16M1__", 1},
    {"atmega32m1", "__AVR_ATmega32M1__", 1},
    {"atmega64m1", "__AVR_ATmega64M1__", 1},
    {"atmega16u4", "__AVR_ATmega16U4__", 1},
    {"atmega32u4", "__AVR_ATmega32U4__", 1},
    {"atmega32u6", "__AVR_ATmega32U6__", 1},
    {"at90usb646", "__AVR_AT90USB646__", 1},
    {"at90usb647", "__AVR_AT90USB647__", 1},
    {"at90scr100", "__AVR_AT90SCR100__", 1},
    {"at94k", "__AVR_AT94K__", 1},
    {"m3000", "__AVR_AT000__", 1},
    {"atmega128", "__AVR_ATmega128__", 2},
    {"atmega128a", "__AVR_ATmega128A__", 2},
    {"atmega1280", "__AVR_ATmega1280__", 2},
    {"atmega1281", "__AVR_ATmega1281__", 2},
    {"atmega1284", "__AVR_ATmega1284__", 2},
    {"atmega1284p", "__AVR_ATmega1284P__", 2},
    {"atmega128rfa1", "__AVR_ATmega128RFA1__", 2},
    {"atmega128rfr2", "__AVR_ATmega128RFR2__", 2},
    {"atmega1284rfr2", "__AVR_ATmega1284RFR2__", 2},
    {"at90can128", "__AVR_AT90CAN128__", 2},
    {"at90usb1286", "__AVR_AT90USB1286__", 2},
    {"at90usb1287", "__AVR_AT90USB1287__", 2},
    {"atmega2560", "__AVR_ATmega2560__", 4},
    {"atmega2561", "__AVR_ATmega2561__", 4},
    {"atmega256rfr2", "__AVR_ATmega256RFR2__", 4},
    {"atmega2564rfr2", "__AVR_ATmega2564RFR2__", 4},
    {"atxmega16a4", "__AVR_ATxmega16A4__", 1},
    {"atxmega16a4u", "__AVR_ATxmega16A4U__", 1},
    {"atxmega16c4", "__AVR_ATxmega16C4__", 1},
    {"atxmega16d4", "__AVR_ATxmega16D4__", 1},
    {"atxmega32a4", "__AVR_ATxmega32A4__", 1},
    {"atxmega32a4u", "__AVR_ATxmega32A4U__", 1},
    {"atxmega32c4", "__AVR_ATxmega32C4__", 1},
    {"atxmega32d4", "__AVR_ATxmega32D4__", 1},
    {"atxmega32e5", "__AVR_ATxmega32E5__", 1},
    {"atxmega16e5", "__AVR_ATxmega16E5__", 1},
    {"atxmega8e5", "__AVR_ATxmega8E5__", 1},
    {"atxmega32x1", "__AVR_ATxmega32X1__", 1},
    {"atxmega64a3", "__AVR_ATxmega64A3__", 1},
    {"atxmega64a3u", "__AVR_ATxmega64A3U__", 1},
    {"atxmega64a4u", "__AVR_ATxmega64A4U__", 1},
    {"atxmega64b1", "__AVR_ATxmega64B1__", 1},
    {"atxmega64b3", "__AVR_ATxmega64B3__", 1},
    {"atxmega64c3", "__AVR_ATxmega64C3__", 1},
    {"atxmega64d3", "__AVR_ATxmega64D3__", 1},
    {"atxmega64d4", "__AVR_ATxmega64D4__", 1},
    {"atxmega64a1", "__AVR_ATxmega64A1__", 1},
    {"atxmega64a1u", "__AVR_ATxmega64A1U__", 1},
    {"atxmega128a3", "__AVR_ATxmega128A3__", 2},
    {"atxmega128a3u", "__AVR_ATxmega128A3U__", 2},
    {"atxmega128b1", "__AVR_ATxmega128B1__", 2},
    {"atxmega128b3", "__AVR_ATxmega128B3__", 2},
    {"atxmega128c3", "__AVR_ATxmega128C3__", 2},
    {"atxmega128d3", "__AVR_ATxmega128D3__", 2},
    {"atxmega128d4", "__AVR_ATxmega128D4__", 2},
    {"atxmega192a3", "__AVR_ATxmega192A3__", 3},
    {"atxmega192a3u", "__AVR_ATxmega192A3U__", 3},
    {"atxmega192c3", "__AVR_ATxmega192C3__", 3},
    {"atxmega192d3", "__AVR_ATxmega192D3__", 3},
    {"atxmega256a3", "__AVR_ATxmega256A3__", 4},
    {"atxmega256a3u", "__AVR_ATxmega256A3U__", 4},
    {"atxmega256a3b", "__AVR_ATxmega256A3B__", 4},
    {"atxmega256a3bu", "__AVR_ATxmega256A3BU__", 4},
    {"atxmega256c3", "__AVR_ATxmega256C3__", 4},
    {"atxmega256d3", "__AVR_ATxmega256D3__", 4},
    {"atxmega384c3", "__AVR_ATxmega384C3__", 6},
    {"atxmega384d3", "__AVR_ATxmega384D3__", 6},
    {"atxmega128a1", "__AVR_ATxmega128A1__", 2},
    {"atxmega128a1u", "__AVR_ATxmega128A1U__", 2},
    {"atxmega128a4u", "__AVR_ATxmega128A4U__", 2},
    {"attiny4", "__AVR_ATtiny4__", 0},
    {"attiny5", "__AVR_ATtiny5__", 0},
    {"attiny9", "__AVR_ATtiny9__", 0},
    {"attiny10", "__AVR_ATtiny10__", 0},
    {"attiny20", "__AVR_ATtiny20__", 0},
    {"attiny40", "__AVR_ATtiny40__", 0},
    {"attiny102", "__AVR_ATtiny102__", 0},
    {"attiny104", "__AVR_ATtiny104__", 0},
    {"attiny202", "__AVR_ATtiny202__", 1},
    {"attiny402", "__AVR_ATtiny402__", 1},
    {"attiny204", "__AVR_ATtiny204__", 1},
    {"attiny404", "__AVR_ATtiny404__", 1},
    {"attiny804", "__AVR_ATtiny804__", 1},
    {"attiny1604", "__AVR_ATtiny1604__", 1},
    {"attiny406", "__AVR_ATtiny406__", 1},
    {"attiny806", "__AVR_ATtiny806__", 1},
    {"attiny1606", "__AVR_ATtiny1606__", 1},
    {"attiny807", "__AVR_ATtiny807__", 1},
    {"attiny1607", "__AVR_ATtiny1607__", 1},
    {"attiny212", "__AVR_ATtiny212__", 1},
    {"attiny412", "__AVR_ATtiny412__", 1},
    {"attiny214", "__AVR_ATtiny214__", 1},
    {"attiny414", "__AVR_ATtiny414__", 1},
    {"attiny814", "__AVR_ATtiny814__", 1},
    {"attiny1614", "__AVR_ATtiny1614__", 1},
    {"attiny416", "__AVR_ATtiny416__", 1},
    {"attiny816", "__AVR_ATtiny816__", 1},
    {"attiny1616", "__AVR_ATtiny1616__", 1},
    {"attiny3216", "__AVR_ATtiny3216__", 1},
    {"attiny417", "__AVR_ATtiny417__", 1},
    {"attiny817", "__AVR_ATtiny817__", 1},
    {"attiny1617", "__AVR_ATtiny1617__", 1},
    {"attiny3217", "__AVR_ATtiny3217__", 1},
};

} // namespace targets
} // namespace clang

static constexpr llvm::StringLiteral ValidFamilyNames[] = {
    "avr1",      "avr2",      "avr25",     "avr3",      "avr31",
    "avr35",     "avr4",      "avr5",      "avr51",     "avr6",
    "avrxmega1", "avrxmega2", "avrxmega3", "avrxmega4", "avrxmega5",
    "avrxmega6", "avrxmega7", "avrtiny"};

bool AVRTargetInfo::isValidCPUName(StringRef Name) const {
  bool IsFamily = llvm::is_contained(ValidFamilyNames, Name);

  bool IsMCU = llvm::any_of(
      AVRMcus, [&](const MCUInfo &Info) { return Info.Name == Name; });
  return IsFamily || IsMCU;
}

void AVRTargetInfo::fillValidCPUList(SmallVectorImpl<StringRef> &Values) const {
  Values.append(std::begin(ValidFamilyNames), std::end(ValidFamilyNames));
  for (const MCUInfo &Info : AVRMcus)
    Values.push_back(Info.Name);
}

void AVRTargetInfo::getTargetDefines(const LangOptions &Opts,
                                     MacroBuilder &Builder) const {
  Builder.defineMacro("AVR");
  Builder.defineMacro("__AVR");
  Builder.defineMacro("__AVR__");
  Builder.defineMacro("__ELF__");

  if (!this->CPU.empty()) {
    auto It = llvm::find_if(
        AVRMcus, [&](const MCUInfo &Info) { return Info.Name == this->CPU; });

    if (It != std::end(AVRMcus)) {
      Builder.defineMacro(It->DefineName);
      if (It->NumFlashBanks >= 1)
        Builder.defineMacro("__flash", "__attribute__((address_space(1)))");
      if (It->NumFlashBanks >= 2)
        Builder.defineMacro("__flash1", "__attribute__((address_space(2)))");
      if (It->NumFlashBanks >= 3)
        Builder.defineMacro("__flash2", "__attribute__((address_space(3)))");
      if (It->NumFlashBanks >= 4)
        Builder.defineMacro("__flash3", "__attribute__((address_space(4)))");
      if (It->NumFlashBanks >= 5)
        Builder.defineMacro("__flash4", "__attribute__((address_space(5)))");
      if (It->NumFlashBanks >= 6)
        Builder.defineMacro("__flash5", "__attribute__((address_space(6)))");
    }
  }
}
