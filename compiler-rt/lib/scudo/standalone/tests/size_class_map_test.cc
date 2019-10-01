//===-- size_class_map_test.cc ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "scudo/standalone/size_class_map.h"
#include "gtest/gtest.h"

template <class SizeClassMap> void testSizeClassMap() {
  typedef SizeClassMap SCMap;
  SCMap::print();
  SCMap::validate();
}

TEST(ScudoSizeClassMapTest, DefaultSizeClassMap) {
  testSizeClassMap<scudo::DefaultSizeClassMap>();
}

TEST(ScudoSizeClassMapTest, SvelteSizeClassMap) {
  testSizeClassMap<scudo::SvelteSizeClassMap>();
}

TEST(ScudoSizeClassMapTest, AndroidSizeClassMap) {
  testSizeClassMap<scudo::AndroidSizeClassMap>();
}

TEST(ScudoSizeClassMapTest, OneClassSizeClassMap) {
  testSizeClassMap<scudo::SizeClassMap<1, 5, 5, 5, 0, 0>>();
}

#if SCUDO_CAN_USE_PRIMARY64
TEST(ScudoSizeClassMapTest, LargeMaxSizeClassMap) {
  testSizeClassMap<scudo::SizeClassMap<3, 4, 8, 63, 128, 16>>();
}
#endif
