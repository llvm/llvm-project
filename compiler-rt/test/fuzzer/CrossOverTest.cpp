// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test for a fuzzer. The fuzzer must find the string
// ABCDEFGHIJ
// We use it as a test for each of CrossOver functionalities
// by passing the following sets of two inputs to it:
// {ABCDE00000, ZZZZZFGHIJ}
// {ABCDEHIJ, ZFG} to specifically test InsertPartOf
// {ABCDE00HIJ, ZFG} to specifically test CopyPartOf
//
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>

static volatile int Sink;
static volatile int *NullPtr;

// A modified jenkins_one_at_a_time_hash initialized by non-zero,
// so that simple_hash(0) != 0. See also
// https://en.wikipedia.org/wiki/Jenkins_hash_function
static uint32_t simple_hash(const uint8_t *Data, size_t Size) {
  uint32_t Hash = 0x12039854;
  for (uint32_t i = 0; i < Size; i++) {
    Hash += Data[i];
    Hash += (Hash << 10);
    Hash ^= (Hash >> 6);
  }
  Hash += (Hash << 3);
  Hash ^= (Hash >> 11);
  Hash += (Hash << 15);
  return Hash;
}

// Don't leave the string in the binary, so that fuzzer don't cheat;
// const char *ABC = "ABCDEFGHIJ";
// static uint32_t ExpectedHash = simple_hash((const uint8_t *)ABC, 10);
static const uint32_t ExpectedHash = 0xe1677acb;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // fprintf(stderr, "ExpectedHash: %x\n", ExpectedHash);
  if (Size == 10 && ExpectedHash == simple_hash(Data, Size))
    *NullPtr = 0;
  // It's UB to read *Data when Size == 0
  if (Size == 0)
    return;
  if (*Data == 'A')
    Sink++;
  if (*Data == 'Z')
    Sink--;
  return 0;
}
