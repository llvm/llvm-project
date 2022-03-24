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
  const int NumFlashBanks; // Set to 0 for the devices do not support LPM/ELPM.
  bool IsTiny; // Set to true for the devices belong to the avrtiny family.
};

// NOTE: This list has been synchronized with gcc-avr 5.4.0 and avr-libc 2.0.0.
static MCUInfo AVRMcus[] = {
    {"at90s1200", "__AVR_AT90S1200__", 0, false},
    {"attiny11", "__AVR_ATtiny11__", 0, false},
    {"attiny12", "__AVR_ATtiny12__", 0, false},
    {"attiny15", "__AVR_ATtiny15__", 0, false},
    {"attiny28", "__AVR_ATtiny28__", 0, false},
    {"at90s2313", "__AVR_AT90S2313__", 1, false},
    {"at90s2323", "__AVR_AT90S2323__", 1, false},
    {"at90s2333", "__AVR_AT90S2333__", 1, false},
    {"at90s2343", "__AVR_AT90S2343__", 1, false},
    {"attiny22", "__AVR_ATtiny22__", 1, false},
    {"attiny26", "__AVR_ATtiny26__", 1, false},
    {"at86rf401", "__AVR_AT86RF401__", 1, false},
    {"at90s4414", "__AVR_AT90S4414__", 1, false},
    {"at90s4433", "__AVR_AT90S4433__", 1, false},
    {"at90s4434", "__AVR_AT90S4434__", 1, false},
    {"at90s8515", "__AVR_AT90S8515__", 1, false},
    {"at90c8534", "__AVR_AT90c8534__", 1, false},
    {"at90s8535", "__AVR_AT90S8535__", 1, false},
    {"ata5272", "__AVR_ATA5272__", 1, false},
    {"ata6616c", "__AVR_ATA6616c__", 1, false},
    {"attiny13", "__AVR_ATtiny13__", 1, false},
    {"attiny13a", "__AVR_ATtiny13A__", 1, false},
    {"attiny2313", "__AVR_ATtiny2313__", 1, false},
    {"attiny2313a", "__AVR_ATtiny2313A__", 1, false},
    {"attiny24", "__AVR_ATtiny24__", 1, false},
    {"attiny24a", "__AVR_ATtiny24A__", 1, false},
    {"attiny4313", "__AVR_ATtiny4313__", 1, false},
    {"attiny44", "__AVR_ATtiny44__", 1, false},
    {"attiny44a", "__AVR_ATtiny44A__", 1, false},
    {"attiny84", "__AVR_ATtiny84__", 1, false},
    {"attiny84a", "__AVR_ATtiny84A__", 1, false},
    {"attiny25", "__AVR_ATtiny25__", 1, false},
    {"attiny45", "__AVR_ATtiny45__", 1, false},
    {"attiny85", "__AVR_ATtiny85__", 1, false},
    {"attiny261", "__AVR_ATtiny261__", 1, false},
    {"attiny261a", "__AVR_ATtiny261A__", 1, false},
    {"attiny441", "__AVR_ATtiny441__", 1, false},
    {"attiny461", "__AVR_ATtiny461__", 1, false},
    {"attiny461a", "__AVR_ATtiny461A__", 1, false},
    {"attiny841", "__AVR_ATtiny841__", 1, false},
    {"attiny861", "__AVR_ATtiny861__", 1, false},
    {"attiny861a", "__AVR_ATtiny861A__", 1, false},
    {"attiny87", "__AVR_ATtiny87__", 1, false},
    {"attiny43u", "__AVR_ATtiny43U__", 1, false},
    {"attiny48", "__AVR_ATtiny48__", 1, false},
    {"attiny88", "__AVR_ATtiny88__", 1, false},
    {"attiny828", "__AVR_ATtiny828__", 1, false},
    {"at43usb355", "__AVR_AT43USB355__", 1, false},
    {"at76c711", "__AVR_AT76C711__", 1, false},
    {"atmega103", "__AVR_ATmega103__", 1, false},
    {"at43usb320", "__AVR_AT43USB320__", 1, false},
    {"attiny167", "__AVR_ATtiny167__", 1, false},
    {"at90usb82", "__AVR_AT90USB82__", 1, false},
    {"at90usb162", "__AVR_AT90USB162__", 1, false},
    {"ata5505", "__AVR_ATA5505__", 1, false},
    {"ata6617c", "__AVR_ATA6617C__", 1, false},
    {"ata664251", "__AVR_ATA664251__", 1, false},
    {"atmega8u2", "__AVR_ATmega8U2__", 1, false},
    {"atmega16u2", "__AVR_ATmega16U2__", 1, false},
    {"atmega32u2", "__AVR_ATmega32U2__", 1, false},
    {"attiny1634", "__AVR_ATtiny1634__", 1, false},
    {"atmega8", "__AVR_ATmega8__", 1, false},
    {"ata6289", "__AVR_ATA6289__", 1, false},
    {"atmega8a", "__AVR_ATmega8A__", 1, false},
    {"ata6285", "__AVR_ATA6285__", 1, false},
    {"ata6286", "__AVR_ATA6286__", 1, false},
    {"ata6612c", "__AVR_ATA6612C__", 1, false},
    {"atmega48", "__AVR_ATmega48__", 1, false},
    {"atmega48a", "__AVR_ATmega48A__", 1, false},
    {"atmega48pa", "__AVR_ATmega48PA__", 1, false},
    {"atmega48pb", "__AVR_ATmega48PB__", 1, false},
    {"atmega48p", "__AVR_ATmega48P__", 1, false},
    {"atmega88", "__AVR_ATmega88__", 1, false},
    {"atmega88a", "__AVR_ATmega88A__", 1, false},
    {"atmega88p", "__AVR_ATmega88P__", 1, false},
    {"atmega88pa", "__AVR_ATmega88PA__", 1, false},
    {"atmega88pb", "__AVR_ATmega88PB__", 1, false},
    {"atmega8515", "__AVR_ATmega8515__", 1, false},
    {"atmega8535", "__AVR_ATmega8535__", 1, false},
    {"atmega8hva", "__AVR_ATmega8HVA__", 1, false},
    {"at90pwm1", "__AVR_AT90PWM1__", 1, false},
    {"at90pwm2", "__AVR_AT90PWM2__", 1, false},
    {"at90pwm2b", "__AVR_AT90PWM2B__", 1, false},
    {"at90pwm3", "__AVR_AT90PWM3__", 1, false},
    {"at90pwm3b", "__AVR_AT90PWM3B__", 1, false},
    {"at90pwm81", "__AVR_AT90PWM81__", 1, false},
    {"ata5702m322", "__AVR_ATA5702M322__", 1, false},
    {"ata5782", "__AVR_ATA5782__", 1, false},
    {"ata5790", "__AVR_ATA5790__", 1, false},
    {"ata5790n", "__AVR_ATA5790N__", 1, false},
    {"ata5791", "__AVR_ATA5791__", 1, false},
    {"ata5795", "__AVR_ATA5795__", 1, false},
    {"ata5831", "__AVR_ATA5831__", 1, false},
    {"ata6613c", "__AVR_ATA6613C__", 1, false},
    {"ata6614q", "__AVR_ATA6614Q__", 1, false},
    {"ata8210", "__AVR_ATA8210__", 1, false},
    {"ata8510", "__AVR_ATA8510__", 1, false},
    {"atmega16", "__AVR_ATmega16__", 1, false},
    {"atmega16a", "__AVR_ATmega16A__", 1, false},
    {"atmega161", "__AVR_ATmega161__", 1, false},
    {"atmega162", "__AVR_ATmega162__", 1, false},
    {"atmega163", "__AVR_ATmega163__", 1, false},
    {"atmega164a", "__AVR_ATmega164A__", 1, false},
    {"atmega164p", "__AVR_ATmega164P__", 1, false},
    {"atmega164pa", "__AVR_ATmega164PA__", 1, false},
    {"atmega165", "__AVR_ATmega165__", 1, false},
    {"atmega165a", "__AVR_ATmega165A__", 1, false},
    {"atmega165p", "__AVR_ATmega165P__", 1, false},
    {"atmega165pa", "__AVR_ATmega165PA__", 1, false},
    {"atmega168", "__AVR_ATmega168__", 1, false},
    {"atmega168a", "__AVR_ATmega168A__", 1, false},
    {"atmega168p", "__AVR_ATmega168P__", 1, false},
    {"atmega168pa", "__AVR_ATmega168PA__", 1, false},
    {"atmega168pb", "__AVR_ATmega168PB__", 1, false},
    {"atmega169", "__AVR_ATmega169__", 1, false},
    {"atmega169a", "__AVR_ATmega169A__", 1, false},
    {"atmega169p", "__AVR_ATmega169P__", 1, false},
    {"atmega169pa", "__AVR_ATmega169PA__", 1, false},
    {"atmega32", "__AVR_ATmega32__", 1, false},
    {"atmega32a", "__AVR_ATmega32A__", 1, false},
    {"atmega323", "__AVR_ATmega323__", 1, false},
    {"atmega324a", "__AVR_ATmega324A__", 1, false},
    {"atmega324p", "__AVR_ATmega324P__", 1, false},
    {"atmega324pa", "__AVR_ATmega324PA__", 1, false},
    {"atmega324pb", "__AVR_ATmega324PB__", 1, false},
    {"atmega325", "__AVR_ATmega325__", 1, false},
    {"atmega325a", "__AVR_ATmega325A__", 1, false},
    {"atmega325p", "__AVR_ATmega325P__", 1, false},
    {"atmega325pa", "__AVR_ATmega325PA__", 1, false},
    {"atmega3250", "__AVR_ATmega3250__", 1, false},
    {"atmega3250a", "__AVR_ATmega3250A__", 1, false},
    {"atmega3250p", "__AVR_ATmega3250P__", 1, false},
    {"atmega3250pa", "__AVR_ATmega3250PA__", 1, false},
    {"atmega328", "__AVR_ATmega328__", 1, false},
    {"atmega328p", "__AVR_ATmega328P__", 1, false},
    {"atmega328pb", "__AVR_ATmega328PB__", 1, false},
    {"atmega329", "__AVR_ATmega329__", 1, false},
    {"atmega329a", "__AVR_ATmega329A__", 1, false},
    {"atmega329p", "__AVR_ATmega329P__", 1, false},
    {"atmega329pa", "__AVR_ATmega329PA__", 1, false},
    {"atmega3290", "__AVR_ATmega3290__", 1, false},
    {"atmega3290a", "__AVR_ATmega3290A__", 1, false},
    {"atmega3290p", "__AVR_ATmega3290P__", 1, false},
    {"atmega3290pa", "__AVR_ATmega3290PA__", 1, false},
    {"atmega406", "__AVR_ATmega406__", 1, false},
    {"atmega64", "__AVR_ATmega64__", 1, false},
    {"atmega64a", "__AVR_ATmega64A__", 1, false},
    {"atmega640", "__AVR_ATmega640__", 1, false},
    {"atmega644", "__AVR_ATmega644__", 1, false},
    {"atmega644a", "__AVR_ATmega644A__", 1, false},
    {"atmega644p", "__AVR_ATmega644P__", 1, false},
    {"atmega644pa", "__AVR_ATmega644PA__", 1, false},
    {"atmega645", "__AVR_ATmega645__", 1, false},
    {"atmega645a", "__AVR_ATmega645A__", 1, false},
    {"atmega645p", "__AVR_ATmega645P__", 1, false},
    {"atmega649", "__AVR_ATmega649__", 1, false},
    {"atmega649a", "__AVR_ATmega649A__", 1, false},
    {"atmega649p", "__AVR_ATmega649P__", 1, false},
    {"atmega6450", "__AVR_ATmega6450__", 1, false},
    {"atmega6450a", "__AVR_ATmega6450A__", 1, false},
    {"atmega6450p", "__AVR_ATmega6450P__", 1, false},
    {"atmega6490", "__AVR_ATmega6490__", 1, false},
    {"atmega6490a", "__AVR_ATmega6490A__", 1, false},
    {"atmega6490p", "__AVR_ATmega6490P__", 1, false},
    {"atmega64rfr2", "__AVR_ATmega64RFR2__", 1, false},
    {"atmega644rfr2", "__AVR_ATmega644RFR2__", 1, false},
    {"atmega16hva", "__AVR_ATmega16HVA__", 1, false},
    {"atmega16hva2", "__AVR_ATmega16HVA2__", 1, false},
    {"atmega16hvb", "__AVR_ATmega16HVB__", 1, false},
    {"atmega16hvbrevb", "__AVR_ATmega16HVBREVB__", 1, false},
    {"atmega32hvb", "__AVR_ATmega32HVB__", 1, false},
    {"atmega32hvbrevb", "__AVR_ATmega32HVBREVB__", 1, false},
    {"atmega64hve", "__AVR_ATmega64HVE__", 1, false},
    {"atmega64hve2", "__AVR_ATmega64HVE2__", 1, false},
    {"at90can32", "__AVR_AT90CAN32__", 1, false},
    {"at90can64", "__AVR_AT90CAN64__", 1, false},
    {"at90pwm161", "__AVR_AT90PWM161__", 1, false},
    {"at90pwm216", "__AVR_AT90PWM216__", 1, false},
    {"at90pwm316", "__AVR_AT90PWM316__", 1, false},
    {"atmega32c1", "__AVR_ATmega32C1__", 1, false},
    {"atmega64c1", "__AVR_ATmega64C1__", 1, false},
    {"atmega16m1", "__AVR_ATmega16M1__", 1, false},
    {"atmega32m1", "__AVR_ATmega32M1__", 1, false},
    {"atmega64m1", "__AVR_ATmega64M1__", 1, false},
    {"atmega16u4", "__AVR_ATmega16U4__", 1, false},
    {"atmega32u4", "__AVR_ATmega32U4__", 1, false},
    {"atmega32u6", "__AVR_ATmega32U6__", 1, false},
    {"at90usb646", "__AVR_AT90USB646__", 1, false},
    {"at90usb647", "__AVR_AT90USB647__", 1, false},
    {"at90scr100", "__AVR_AT90SCR100__", 1, false},
    {"at94k", "__AVR_AT94K__", 1, false},
    {"m3000", "__AVR_AT000__", 1, false},
    {"atmega128", "__AVR_ATmega128__", 2, false},
    {"atmega128a", "__AVR_ATmega128A__", 2, false},
    {"atmega1280", "__AVR_ATmega1280__", 2, false},
    {"atmega1281", "__AVR_ATmega1281__", 2, false},
    {"atmega1284", "__AVR_ATmega1284__", 2, false},
    {"atmega1284p", "__AVR_ATmega1284P__", 2, false},
    {"atmega128rfa1", "__AVR_ATmega128RFA1__", 2, false},
    {"atmega128rfr2", "__AVR_ATmega128RFR2__", 2, false},
    {"atmega1284rfr2", "__AVR_ATmega1284RFR2__", 2, false},
    {"at90can128", "__AVR_AT90CAN128__", 2, false},
    {"at90usb1286", "__AVR_AT90USB1286__", 2, false},
    {"at90usb1287", "__AVR_AT90USB1287__", 2, false},
    {"atmega2560", "__AVR_ATmega2560__", 4, false},
    {"atmega2561", "__AVR_ATmega2561__", 4, false},
    {"atmega256rfr2", "__AVR_ATmega256RFR2__", 4, false},
    {"atmega2564rfr2", "__AVR_ATmega2564RFR2__", 4, false},
    {"atxmega16a4", "__AVR_ATxmega16A4__", 1, false},
    {"atxmega16a4u", "__AVR_ATxmega16A4U__", 1, false},
    {"atxmega16c4", "__AVR_ATxmega16C4__", 1, false},
    {"atxmega16d4", "__AVR_ATxmega16D4__", 1, false},
    {"atxmega32a4", "__AVR_ATxmega32A4__", 1, false},
    {"atxmega32a4u", "__AVR_ATxmega32A4U__", 1, false},
    {"atxmega32c3", "__AVR_ATxmega32C3__", 1, false},
    {"atxmega32c4", "__AVR_ATxmega32C4__", 1, false},
    {"atxmega32d3", "__AVR_ATxmega32D3__", 1, false},
    {"atxmega32d4", "__AVR_ATxmega32D4__", 1, false},
    {"atxmega32e5", "__AVR_ATxmega32E5__", 1, false},
    {"atxmega16e5", "__AVR_ATxmega16E5__", 1, false},
    {"atxmega8e5", "__AVR_ATxmega8E5__", 1, false},
    {"atxmega64a3", "__AVR_ATxmega64A3__", 1, false},
    {"atxmega64a3u", "__AVR_ATxmega64A3U__", 1, false},
    {"atxmega64a4u", "__AVR_ATxmega64A4U__", 1, false},
    {"atxmega64b1", "__AVR_ATxmega64B1__", 1, false},
    {"atxmega64b3", "__AVR_ATxmega64B3__", 1, false},
    {"atxmega64c3", "__AVR_ATxmega64C3__", 1, false},
    {"atxmega64d3", "__AVR_ATxmega64D3__", 1, false},
    {"atxmega64d4", "__AVR_ATxmega64D4__", 1, false},
    {"atxmega64a1", "__AVR_ATxmega64A1__", 1, false},
    {"atxmega64a1u", "__AVR_ATxmega64A1U__", 1, false},
    {"atxmega128a3", "__AVR_ATxmega128A3__", 2, false},
    {"atxmega128a3u", "__AVR_ATxmega128A3U__", 2, false},
    {"atxmega128b1", "__AVR_ATxmega128B1__", 2, false},
    {"atxmega128b3", "__AVR_ATxmega128B3__", 2, false},
    {"atxmega128c3", "__AVR_ATxmega128C3__", 2, false},
    {"atxmega128d3", "__AVR_ATxmega128D3__", 2, false},
    {"atxmega128d4", "__AVR_ATxmega128D4__", 2, false},
    {"atxmega192a3", "__AVR_ATxmega192A3__", 3, false},
    {"atxmega192a3u", "__AVR_ATxmega192A3U__", 3, false},
    {"atxmega192c3", "__AVR_ATxmega192C3__", 3, false},
    {"atxmega192d3", "__AVR_ATxmega192D3__", 3, false},
    {"atxmega256a3", "__AVR_ATxmega256A3__", 4, false},
    {"atxmega256a3u", "__AVR_ATxmega256A3U__", 4, false},
    {"atxmega256a3b", "__AVR_ATxmega256A3B__", 4, false},
    {"atxmega256a3bu", "__AVR_ATxmega256A3BU__", 4, false},
    {"atxmega256c3", "__AVR_ATxmega256C3__", 4, false},
    {"atxmega256d3", "__AVR_ATxmega256D3__", 4, false},
    {"atxmega384c3", "__AVR_ATxmega384C3__", 6, false},
    {"atxmega384d3", "__AVR_ATxmega384D3__", 6, false},
    {"atxmega128a1", "__AVR_ATxmega128A1__", 2, false},
    {"atxmega128a1u", "__AVR_ATxmega128A1U__", 2, false},
    {"atxmega128a4u", "__AVR_ATxmega128A4U__", 2, false},
    {"attiny4", "__AVR_ATtiny4__", 0, true},
    {"attiny5", "__AVR_ATtiny5__", 0, true},
    {"attiny9", "__AVR_ATtiny9__", 0, true},
    {"attiny10", "__AVR_ATtiny10__", 0, true},
    {"attiny20", "__AVR_ATtiny20__", 0, true},
    {"attiny40", "__AVR_ATtiny40__", 0, true},
    {"attiny102", "__AVR_ATtiny102__", 0, true},
    {"attiny104", "__AVR_ATtiny104__", 0, true},
    {"attiny202", "__AVR_ATtiny202__", 1, false},
    {"attiny402", "__AVR_ATtiny402__", 1, false},
    {"attiny204", "__AVR_ATtiny204__", 1, false},
    {"attiny404", "__AVR_ATtiny404__", 1, false},
    {"attiny804", "__AVR_ATtiny804__", 1, false},
    {"attiny1604", "__AVR_ATtiny1604__", 1, false},
    {"attiny406", "__AVR_ATtiny406__", 1, false},
    {"attiny806", "__AVR_ATtiny806__", 1, false},
    {"attiny1606", "__AVR_ATtiny1606__", 1, false},
    {"attiny807", "__AVR_ATtiny807__", 1, false},
    {"attiny1607", "__AVR_ATtiny1607__", 1, false},
    {"attiny212", "__AVR_ATtiny212__", 1, false},
    {"attiny412", "__AVR_ATtiny412__", 1, false},
    {"attiny214", "__AVR_ATtiny214__", 1, false},
    {"attiny414", "__AVR_ATtiny414__", 1, false},
    {"attiny814", "__AVR_ATtiny814__", 1, false},
    {"attiny1614", "__AVR_ATtiny1614__", 1, false},
    {"attiny416", "__AVR_ATtiny416__", 1, false},
    {"attiny816", "__AVR_ATtiny816__", 1, false},
    {"attiny1616", "__AVR_ATtiny1616__", 1, false},
    {"attiny3216", "__AVR_ATtiny3216__", 1, false},
    {"attiny417", "__AVR_ATtiny417__", 1, false},
    {"attiny817", "__AVR_ATtiny817__", 1, false},
    {"attiny1617", "__AVR_ATtiny1617__", 1, false},
    {"attiny3217", "__AVR_ATtiny3217__", 1, false},
    {"attiny1624", "__AVR_ATtiny1624__", 1, false},
    {"attiny1626", "__AVR_ATtiny1626__", 1, false},
    {"attiny1627", "__AVR_ATtiny1627__", 1, false},
    {"atmega808", "__AVR_ATmega808__", 1, false},
    {"atmega809", "__AVR_ATmega809__", 1, false},
    {"atmega1608", "__AVR_ATmega1608__", 1, false},
    {"atmega1609", "__AVR_ATmega1609__", 1, false},
    {"atmega3208", "__AVR_ATmega3208__", 1, false},
    {"atmega3209", "__AVR_ATmega3209__", 1, false},
    {"atmega4808", "__AVR_ATmega4808__", 1, false},
    {"atmega4809", "__AVR_ATmega4809__", 1, false},
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

bool AVRTargetInfo::setCPU(const std::string &Name) {
  // Set the ABI and CPU fields if parameter Name is a family name.
  if (llvm::is_contained(ValidFamilyNames, Name)) {
    CPU = Name;
    ABI = Name == "avrtiny" ? "avrtiny" : "avr";
    return true;
  }

  // Set the ABI field if parameter Name is a device name.
  auto It = llvm::find_if(
      AVRMcus, [&](const MCUInfo &Info) { return Info.Name == Name; });
  if (It != std::end(AVRMcus)) {
    CPU = Name;
    ABI = It->IsTiny ? "avrtiny" : "avr";
    return true;
  }

  // Parameter Name is neither valid family name nor valid device name.
  return false;
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
