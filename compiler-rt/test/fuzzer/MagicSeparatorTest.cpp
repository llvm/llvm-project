// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer.
// This is a sample fuzz target for a custom serialization format that uses
// a magic separator to split the input into several independent buffers.
// The fuzzer must find the input consisting of 2 subinputs: "Fuzz" and "me".
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <vector>

// Splits [data,data+size) into a vector of strings using a "magic" Separator.
std::vector<std::vector<uint8_t>> SplitInput(const uint8_t *Data, size_t Size,
                                     const uint8_t *Separator,
                                     size_t SeparatorSize) {
  std::vector<std::vector<uint8_t>> Res;
  assert(SeparatorSize > 0);
  auto Beg = Data;
  auto End = Data + Size;
  // Using memmem here. std::search may be harder for libFuzzer today.
  while (const uint8_t *Pos = (const uint8_t *)memmem(Beg, End - Beg,
                                     Separator, SeparatorSize)) {
    Res.push_back({Beg, Pos});
    Beg = Pos + SeparatorSize;
  }
  if (Beg < End)
    Res.push_back({Beg, End});
  return Res;
}

static volatile int *Nil = nullptr;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 10) return 0;  // To make the test quick.
  const uint8_t Separator[] = {0xDE, 0xAD, 0xBE, 0xEF};
  auto Inputs = SplitInput(Data, Size, Separator, sizeof(Separator));
  std::vector<uint8_t> Fuzz({'F', 'u', 'z', 'z'});
  std::vector<uint8_t> Me({'m', 'e'});
  if (Inputs.size() == 2 && Inputs[0] == Fuzz && Inputs[1] == Me)
    *Nil = 42;  // crash.
  return 0;
}

