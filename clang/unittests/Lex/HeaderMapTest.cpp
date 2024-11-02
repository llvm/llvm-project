//===- unittests/Lex/HeaderMapTest.cpp - HeaderMap tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include "HeaderMapTestUtils.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace clang;
using namespace llvm;
using namespace clang::test;

namespace {

TEST(HeaderMapTest, checkHeaderEmpty) {
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(
      *MemoryBuffer::getMemBufferCopy("", "empty"), NeedsSwap));
  ASSERT_FALSE(HeaderMapImpl::checkHeader(
      *MemoryBuffer::getMemBufferCopy("", "empty"), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderMagic) {
  HMapFileMock<1, 1> File;
  File.init();
  File.Header.Magic = 0;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderReserved) {
  HMapFileMock<1, 1> File;
  File.init();
  File.Header.Reserved = 1;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderVersion) {
  HMapFileMock<1, 1> File;
  File.init();
  ++File.Header.Version;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderValidButEmpty) {
  HMapFileMock<1, 1> File;
  File.init();
  bool NeedsSwap;
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_FALSE(NeedsSwap);

  File.swapBytes();
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_TRUE(NeedsSwap);
}

TEST(HeaderMapTest, checkHeader3Buckets) {
  HMapFileMock<3, 1> File;
  ASSERT_EQ(3 * sizeof(HMapBucket), sizeof(File.Buckets));

  File.init();
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeader0Buckets) {
  // Create with 1 bucket to avoid 0-sized arrays.
  HMapFileMock<1, 1> File;
  File.init();
  File.Header.NumBuckets = 0;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderNotEnoughBuckets) {
  HMapFileMock<1, 1> File;
  File.init();
  File.Header.NumBuckets = 8;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, lookupFilename) {
  typedef HMapFileMock<2, 7> FileTy;
  FileTy File;
  File.init();

  HMapFileMockMaker<FileTy> Maker(File);
  auto a = Maker.addString("a");
  auto b = Maker.addString("b");
  auto c = Maker.addString("c");
  Maker.addBucket("a", a, b, c);

  bool NeedsSwap;
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_FALSE(NeedsSwap);
  HeaderMapImpl Map(File.getBuffer(), NeedsSwap);

  SmallString<8> DestPath;
  ASSERT_EQ("bc", Map.lookupFilename("a", DestPath));
}

template <class FileTy, class PaddingTy> struct PaddedFile {
  FileTy File;
  PaddingTy Padding;
};

TEST(HeaderMapTest, lookupFilenameTruncatedSuffix) {
  typedef HMapFileMock<2, 64 - sizeof(HMapHeader) - 2 * sizeof(HMapBucket)>
      FileTy;
  static_assert(std::is_standard_layout_v<FileTy>, "Expected standard layout");
  static_assert(sizeof(FileTy) == 64, "check the math");
  PaddedFile<FileTy, uint64_t> P;
  auto &File = P.File;
  auto &Padding = P.Padding;
  File.init();

  HMapFileMockMaker<FileTy> Maker(File);
  auto a = Maker.addString("a");
  auto b = Maker.addString("b");
  auto c = Maker.addString("c");
  Maker.addBucket("a", a, b, c);

  // Add 'x' characters to cause an overflow into Padding.
  ASSERT_EQ('c', File.Bytes[5]);
  for (unsigned I = 6; I < sizeof(File.Bytes); ++I) {
    ASSERT_EQ(0, File.Bytes[I]);
    File.Bytes[I] = 'x';
  }
  Padding = 0xffffffff; // Padding won't stop it either.

  bool NeedsSwap;
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_FALSE(NeedsSwap);
  HeaderMapImpl Map(File.getBuffer(), NeedsSwap);

  // The string for "c" runs to the end of File.  Check that the suffix
  // ("cxxxx...") is detected as truncated, and an empty string is returned.
  SmallString<24> DestPath;
  ASSERT_EQ("", Map.lookupFilename("a", DestPath));
}

TEST(HeaderMapTest, lookupFilenameTruncatedPrefix) {
  typedef HMapFileMock<2, 64 - sizeof(HMapHeader) - 2 * sizeof(HMapBucket)>
      FileTy;
  static_assert(std::is_standard_layout_v<FileTy>, "Expected standard layout");
  static_assert(sizeof(FileTy) == 64, "check the math");
  PaddedFile<FileTy, uint64_t> P;
  auto &File = P.File;
  auto &Padding = P.Padding;
  File.init();

  HMapFileMockMaker<FileTy> Maker(File);
  auto a = Maker.addString("a");
  auto c = Maker.addString("c");
  auto b = Maker.addString("b"); // Store the prefix last.
  Maker.addBucket("a", a, b, c);

  // Add 'x' characters to cause an overflow into Padding.
  ASSERT_EQ('b', File.Bytes[5]);
  for (unsigned I = 6; I < sizeof(File.Bytes); ++I) {
    ASSERT_EQ(0, File.Bytes[I]);
    File.Bytes[I] = 'x';
  }
  Padding = 0xffffffff; // Padding won't stop it either.

  bool NeedsSwap;
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_FALSE(NeedsSwap);
  HeaderMapImpl Map(File.getBuffer(), NeedsSwap);

  // The string for "b" runs to the end of File.  Check that the prefix
  // ("bxxxx...") is detected as truncated, and an empty string is returned.
  SmallString<24> DestPath;
  ASSERT_EQ("", Map.lookupFilename("a", DestPath));
}

} // end namespace
