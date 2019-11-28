//===-- bytemap_test.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tests/scudo_unit_test.h"

#include "bytemap.h"

#include <pthread.h>
#include <string.h>

template <typename T> void testMap(T &Map, scudo::uptr Size) {
  Map.init();
  for (scudo::uptr I = 0; I < Size; I += 7)
    Map.set(I, (I % 100) + 1);
  for (scudo::uptr J = 0; J < Size; J++) {
    if (J % 7)
      EXPECT_EQ(Map[J], 0);
    else
      EXPECT_EQ(Map[J], (J % 100) + 1);
  }
}

TEST(ScudoByteMapTest, FlatByteMap) {
  const scudo::uptr Size = 1U << 10;
  scudo::FlatByteMap<Size> Map;
  testMap(Map, Size);
  Map.unmapTestOnly();
}

TEST(ScudoByteMapTest, TwoLevelByteMap) {
  const scudo::uptr Size1 = 1U << 6, Size2 = 1U << 12;
  scudo::TwoLevelByteMap<Size1, Size2> Map;
  testMap(Map, Size1 * Size2);
  Map.unmapTestOnly();
}

using TestByteMap = scudo::TwoLevelByteMap<1U << 12, 1U << 13>;

struct TestByteMapParam {
  TestByteMap *Map;
  scudo::uptr Shard;
  scudo::uptr NumberOfShards;
};

void *populateByteMap(void *Param) {
  TestByteMapParam *P = reinterpret_cast<TestByteMapParam *>(Param);
  for (scudo::uptr I = P->Shard; I < P->Map->size(); I += P->NumberOfShards) {
    scudo::u8 V = static_cast<scudo::u8>((I % 100) + 1);
    P->Map->set(I, V);
    EXPECT_EQ((*P->Map)[I], V);
  }
  return 0;
}

TEST(ScudoByteMapTest, ThreadedTwoLevelByteMap) {
  TestByteMap Map;
  Map.init();
  static const scudo::uptr NumberOfThreads = 16U;
  pthread_t T[NumberOfThreads];
  TestByteMapParam P[NumberOfThreads];
  for (scudo::uptr I = 0; I < NumberOfThreads; I++) {
    P[I].Map = &Map;
    P[I].Shard = I;
    P[I].NumberOfShards = NumberOfThreads;
    pthread_create(&T[I], 0, populateByteMap, &P[I]);
  }
  for (scudo::uptr I = 0; I < NumberOfThreads; I++)
    pthread_join(T[I], 0);
  Map.unmapTestOnly();
}
