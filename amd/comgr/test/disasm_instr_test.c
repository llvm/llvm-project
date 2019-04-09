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

const int expectedUserData;

void checkUserData(void *userData) {
  if (userData != (void *)&expectedUserData)
    fail("user_data changed");
}

const char *skipspace(const char *s) {
  while (isspace(*s))
    ++s;
  return s;
}

size_t strlenWithoutTrailingWhitespace(const char *s) {
  size_t i = strlen(s);
  while (i && isspace(s[--i]))
    ;
  return i + 1;
}

const char program[] = {
    '\x02', '\x00', '\x06', '\xC0', '\x00', '\x00', '\x00', '\x00', '\x7f',
    '\xC0', '\x8c', '\xbf', '\x00', '\x80', '\x12', '\xbf', '\x05', '\x00',
    '\x85', '\xbf', '\x00', '\x02', '\x00', '\x7e', '\xc0', '\x02', '\x04',
    '\x7e', '\x01', '\x02', '\x02', '\x7e', '\x00', '\x80', '\x70', '\xdc',
    '\x00', '\x02', '\x7f', '\x00', '\x00', '\x00', '\x81', '\xbf',
};

const char *instructions[] = {
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
const size_t instructionsLen = sizeof(instructions) / sizeof(*instructions);
size_t instructionsIdx = 0;
const size_t brInstructionIdx = 3;
const size_t brInstructionAddr = 40;

uint64_t readMemoryCallback(uint64_t from, char *to, uint64_t size,
                            void *userData) {
  checkUserData(userData);
  if (from >= sizeof(program))
    return 0;
  if (from + size > sizeof(program))
    size = sizeof(program) - from;
  memcpy(to, program + from, size);
  return size;
}

void printInstructionCallback(const char *instruction, void *userData) {
  checkUserData(userData);
  if (instructionsIdx == instructionsLen)
    fail("too many instructions");
  const char *expected = skipspace(instructions[instructionsIdx++]);
  const char *actual = skipspace(instruction);
  if (strncmp(expected, actual, strlenWithoutTrailingWhitespace(actual)))
    fail("incorrect instruction: expected '%s', actual '%s'", expected, actual);
}

void printAddressCallback(uint64_t address, void *userData) {
  checkUserData(userData);
  size_t actualIdx = instructionsIdx - 1;
  if (actualIdx != brInstructionIdx)
    fail("absolute address resolved for instruction index %zu, expected index "
         "%zu",
         instructionsIdx, brInstructionIdx);
  if (address != brInstructionAddr)
    fail("incorrect absolute address %u resolved for instruction index %zu, "
         "expected %u",
         address, actualIdx, brInstructionAddr);
}

int main(int argc, char *argv[]) {
  amd_comgr_status_t status;

  amd_comgr_disassembly_info_t disassemblyInfo;

  status = amd_comgr_create_disassembly_info(
      "amdgcn-amd-amdhsa--gfx900", &readMemoryCallback,
      &printInstructionCallback, &printAddressCallback, &disassemblyInfo);
  checkError(status, "amd_comgr_create_disassembly_info");

  uint64_t addr = 0;
  uint64_t size = 0;
  while (status == AMD_COMGR_STATUS_SUCCESS && addr < sizeof(program)) {
    status = amd_comgr_disassemble_instruction(
        disassemblyInfo, addr, (void *)&expectedUserData, &size);
    checkError(status, "amd_comgr_disassemble_instruction");
    addr += size;
  }

  if (instructionsIdx != instructionsLen)
    fail("too few instructions\n");

  addr = sizeof(program) - 1;
  size = 0;
  status = amd_comgr_disassemble_instruction(disassemblyInfo, addr,
                                             (void *)&expectedUserData, &size);
  if (status != AMD_COMGR_STATUS_ERROR)
    fail("successfully disassembled invalid instruction encoding");

  status = amd_comgr_destroy_disassembly_info(disassemblyInfo);
  checkError(status, "amd_comgr_destroy_disassembly_info");

  return EXIT_SUCCESS;
}
