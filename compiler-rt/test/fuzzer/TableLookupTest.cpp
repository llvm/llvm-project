// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Make sure the fuzzer eventually finds all possible values of a variable
// within a range.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

const size_t N = 1 << 12;

// Define an array of counters that will be understood by libFuzzer
// as extra coverage signal. The array must be:
//  * uint8_t
//  * in the section named __libfuzzer_extra_counters.
// The target code may declare more than one such array.
//
// Use either `Counters[Idx] = 1` or `Counters[Idx]++;`
// depending on whether multiple occurrences of the event 'Idx'
// is important to distinguish from one occurrence.
#ifdef __linux__
__attribute__((section("__libfuzzer_extra_counters")))
#endif
static uint8_t Counters[N];

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  static std::set<uint16_t> SeenIdx;
  if (Size != 4) return 0;
  uint32_t Idx;
  memcpy(&Idx, Data, 4);
  Idx %= N;
  assert(Counters[Idx] == 0);  // libFuzzer should reset these between the runs.
  // Or Counters[Idx]=1 if we don't care how many times this happened.
  Counters[Idx]++;
  SeenIdx.insert(Idx);
  if (SeenIdx.size() == N) {
    fprintf(stderr, "BINGO: found all values\n");
    abort();
  }
  return 0;
}
