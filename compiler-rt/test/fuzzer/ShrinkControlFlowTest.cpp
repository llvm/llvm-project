// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test that we can find the minimal item in the corpus (3 bytes: "FUZ").
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static volatile int Sink;

void Foo() {
  Sink++;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  int8_t Ids[256];
  memset(Ids, -1, sizeof(Ids));
  for (size_t i = 0; i < Size; i++)
    if (Ids[Data[i]] == -1)
      Ids[Data[i]] = i;
  int F = Ids[(unsigned char)'F'];
  int U = Ids[(unsigned char)'U'];
  int Z = Ids[(unsigned char)'Z'];
  if (F >= 0 && U > F && Z > U) {
    Foo();
  }
  return 0;
}

