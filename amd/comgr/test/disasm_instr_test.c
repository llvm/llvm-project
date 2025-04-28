/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "amd_comgr.h"
#include "common.h"
#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int ExpectedUserData;

void checkUserData(void *UserData) {
  if (UserData != (void *)&ExpectedUserData) {
    fail("user_data changed");
  }
}

const char *skipspace(const char *S) {
  while (isspace(*S)) {
    ++S;
  }
  return S;
}

size_t strlenWithoutTrailingWhitespace(const char *S) {
  size_t I = strlen(S);
  while (I && isspace(S[--I])) {
    ;
  }
  return I + 1;
}

const char Program[] = {
    '\x02', '\x00', '\x06', '\xC0', '\x00', '\x00', '\x00', '\x00', '\x7f',
    '\xC0', '\x8c', '\xbf', '\x00', '\x80', '\x12', '\xbf', '\x05', '\x00',
    '\x85', '\xbf', '\x00', '\x02', '\x00', '\x7e', '\xc0', '\x02', '\x04',
    '\x7e', '\x01', '\x02', '\x02', '\x7e', '\x00', '\x80', '\x70', '\xdc',
    '\x00', '\x02', '\x7f', '\x00', '\x00', '\x00', '\x81', '\xbf',
};

const char *Instructions[] = {
    "s_load_dwordx2 s[0:1], s[4:5], 0x0",
    "s_waitcnt lgkmcnt(0)",
    "s_cmp_eq_u64 s[0:1], 0",
    "s_cbranch_scc1 5",
    "v_mov_b32_e32 v0, s0",
    "v_mov_b32_e32 v2, 64",
    "v_mov_b32_e32 v1, s1",
    "global_store_dword v[0:1], v2, off",
    "s_endpgm",
};
const size_t InstructionsLen = sizeof(Instructions) / sizeof(*Instructions);
size_t InstructionsIdx = 0;
const size_t BrInstructionIdx = 3;
const size_t BrInstructionAddr = 40;

uint64_t readMemoryCallback(uint64_t From, char *To, uint64_t Size,
                            void *UserData) {
  checkUserData(UserData);
  if (From >= sizeof(Program)) {
    return 0;
  }
  if (From + Size > sizeof(Program)) {
    Size = sizeof(Program) - From;
  }
  memcpy(To, Program + From, Size);
  return Size;
}

void printInstructionCallback(const char *Instruction, void *UserData) {
  checkUserData(UserData);
  if (InstructionsIdx == InstructionsLen) {
    fail("too many instructions");
  }
  const char *Expected = skipspace(Instructions[InstructionsIdx++]);
  const char *Actual = skipspace(Instruction);
  if (strncmp(Expected, Actual, strlenWithoutTrailingWhitespace(Actual))) {
    fail("incorrect instruction: expected '%s', actual '%s'", Expected, Actual);
  }
}

void printAddressCallback(uint64_t Address, void *UserData) {
  checkUserData(UserData);
  size_t ActualIdx = InstructionsIdx - 1;
  if (ActualIdx != BrInstructionIdx) {
    fail("absolute address resolved for instruction index %zu, expected index "
         "%zu",
         InstructionsIdx, BrInstructionIdx);
  }
  if (Address != BrInstructionAddr) {
    fail("incorrect absolute address %u resolved for instruction index %zu, "
         "expected %u",
         Address, ActualIdx, BrInstructionAddr);
  }
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t Status;

  amd_comgr_disassembly_info_t DisassemblyInfo;

  Status = amd_comgr_create_disassembly_info(
      "amdgcn-amd-amdhsa--gfx900", &readMemoryCallback,
      &printInstructionCallback, &printAddressCallback, &DisassemblyInfo);
  checkError(Status, "amd_comgr_create_disassembly_info");

  uint64_t Addr = 0;
  uint64_t Size = 0;
  while (Status == AMD_COMGR_STATUS_SUCCESS && Addr < sizeof(Program)) {
    Status = amd_comgr_disassemble_instruction(
        DisassemblyInfo, Addr, (void *)&ExpectedUserData, &Size);
    checkError(Status, "amd_comgr_disassemble_instruction");
    Addr += Size;
  }

  if (InstructionsIdx != InstructionsLen) {
    fail("too few instructions\n");
  }

  Addr = sizeof(Program) - 1;
  Size = 0;
  Status = amd_comgr_disassemble_instruction(DisassemblyInfo, Addr,
                                             (void *)&ExpectedUserData, &Size);
  if (Status != AMD_COMGR_STATUS_ERROR) {
    fail("successfully disassembled invalid instruction encoding");
  }

  Status = amd_comgr_destroy_disassembly_info(DisassemblyInfo);
  checkError(Status, "amd_comgr_destroy_disassembly_info");

  return EXIT_SUCCESS;
}
