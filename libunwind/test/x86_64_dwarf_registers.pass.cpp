//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{x86_64-.+}}
// UNSUPPORTED: target={{.*-windows.*}}
// ADDITIONAL_COMPILE_FLAGS: -D_LIBUNWIND_IS_NATIVE_ONLY

// Disable internal tracing, which depends on private library symbols.
#ifndef NDEBUG
#define NDEBUG
#endif
#include "../src/AddressSpace.hpp"
#include "../src/DwarfParser.hpp"
#undef NDEBUG
#include <assert.h>

using namespace libunwind;

int main(int, char **) {
  using Parser = CFI_Parser<LocalAddressSpace>;
  LocalAddressSpace addressSpace;
  Parser::CIE_Info cie = {};
  cie.codeAlignFactor = 1;
  cie.dataAlignFactor = -8;
  Parser::FDE_Info fde = {};
  const uint8_t instructions[] = {
      DW_CFA_offset_extended, 49, 1, DW_CFA_offset_extended, 50, 2,
      DW_CFA_offset_extended, 51, 3, DW_CFA_offset_extended, 52, 4,
      DW_CFA_offset_extended, 53, 5, DW_CFA_offset_extended, 54, 6,
      DW_CFA_offset_extended, 55, 7, DW_CFA_offset_extended, 58, 8,
      DW_CFA_offset_extended, 59, 9};
  fde.fdeInstructions = reinterpret_cast<uintptr_t>(instructions);
  fde.fdeStart = fde.fdeInstructions;
  fde.fdeLength = sizeof(instructions);
  Parser::PrologInfo prolog;
  assert(Parser::parseFDEInstructions<Registers_x86_64>(
      addressSpace, fde, cie, 1, REGISTERS_X86_64, &prolog));
  const int regs[] = {49, 50, 51, 52, 53, 54, 55, 58, 59};
  Registers_x86_64 registers;
  for (unsigned i = 0; i != sizeof(regs) / sizeof(regs[0]); ++i) {
    assert(prolog.savedRegisters[regs[i]].location == Parser::kRegisterInCFA);
    assert(prolog.savedRegisters[regs[i]].value == -8 * int(i + 1));
    // Also exercise the base fields on hosts without native base access.
    registers.setRegister(regs[i], i + 1);
    assert(registers.getRegister(regs[i]) == i + 1);
  }
  const uint8_t invalid[] = {DW_CFA_offset_extended, 60, 1};
  fde.fdeInstructions = reinterpret_cast<uintptr_t>(invalid);
  fde.fdeStart = fde.fdeInstructions;
  fde.fdeLength = sizeof(invalid);
  assert(!Parser::parseFDEInstructions<Registers_x86_64>(
      addressSpace, fde, cie, 1, REGISTERS_X86_64, &prolog));
  return 0;
}
