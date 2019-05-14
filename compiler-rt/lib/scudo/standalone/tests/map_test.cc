//===-- map_test.cc ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.h"

#include "gtest/gtest.h"

#include <string.h>

static const char *MappingName = "scudo:test";

TEST(ScudoMapTest, MapNoAccessUnmap) {
  const scudo::uptr Size = 4 * scudo::getPageSizeCached();
  scudo::MapPlatformData Data = {};
  void *P = scudo::map(nullptr, Size, MappingName, MAP_NOACCESS, &Data);
  EXPECT_NE(P, nullptr);
  EXPECT_DEATH(memset(P, 0xaa, Size), "");
  scudo::unmap(P, Size, UNMAP_ALL, &Data);
}

TEST(ScudoMapTest, MapUnmap) {
  const scudo::uptr Size = 4 * scudo::getPageSizeCached();
  scudo::MapPlatformData Data = {};
  void *P = scudo::map(nullptr, Size, MappingName, 0, &Data);
  EXPECT_NE(P, nullptr);
  memset(P, 0xaa, Size);
  scudo::unmap(P, Size, 0, &Data);
  EXPECT_DEATH(memset(P, 0xbb, Size), "");
}

TEST(ScudoMapTest, MapWithGuardUnmap) {
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  const scudo::uptr Size = 4 * PageSize;
  scudo::MapPlatformData Data = {};
  void *P = scudo::map(nullptr, Size + 2 * PageSize, MappingName, MAP_NOACCESS,
                       &Data);
  EXPECT_NE(P, nullptr);
  void *Q =
      reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(P) + PageSize);
  EXPECT_EQ(scudo::map(Q, Size, MappingName, 0, &Data), Q);
  memset(Q, 0xaa, Size);
  EXPECT_DEATH(memset(Q, 0xaa, Size + 1), "");
  scudo::unmap(P, Size + 2 * PageSize, UNMAP_ALL, &Data);
}

TEST(ScudoMapTest, MapGrowUnmap) {
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  const scudo::uptr Size = 4 * PageSize;
  scudo::MapPlatformData Data = {};
  void *P = scudo::map(nullptr, Size, MappingName, MAP_NOACCESS, &Data);
  EXPECT_NE(P, nullptr);
  void *Q =
      reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(P) + PageSize);
  EXPECT_EQ(scudo::map(Q, PageSize, MappingName, 0, &Data), Q);
  memset(Q, 0xaa, PageSize);
  Q = reinterpret_cast<void *>(reinterpret_cast<scudo::uptr>(Q) + PageSize);
  EXPECT_EQ(scudo::map(Q, PageSize, MappingName, 0, &Data), Q);
  memset(Q, 0xbb, PageSize);
  scudo::unmap(P, Size, UNMAP_ALL, &Data);
}
