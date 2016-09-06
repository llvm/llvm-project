//===- unittests/Lex/HeaderMapTest.cpp - HeaderMap tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------===//

#include "clang/Basic/CharInfo.h"
#include "clang/Lex/HeaderMap.h"
#include "clang/Lex/HeaderMapTypes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SwapByteOrder.h"
#include "gtest/gtest.h"
#include <cassert>
#include <type_traits>

using namespace clang;
using namespace llvm;

namespace {

// Lay out a header file for testing.
template <unsigned NumBuckets, unsigned NumBytes> struct MapFile {
  HMapHeader Header;
  HMapBucket Buckets[NumBuckets];
  unsigned char Bytes[NumBytes];

  void init() {
    memset(this, 0, sizeof(MapFile));
    Header.Magic = HMAP_HeaderMagicNumber;
    Header.Version = HMAP_HeaderVersion;
    Header.NumBuckets = NumBuckets;
    Header.StringsOffset = sizeof(Header) + sizeof(Buckets);
  }

  void swapBytes() {
    using llvm::sys::getSwappedBytes;
    Header.Magic = getSwappedBytes(Header.Magic);
    Header.Version = getSwappedBytes(Header.Version);
    Header.NumBuckets = getSwappedBytes(Header.NumBuckets);
    Header.StringsOffset = getSwappedBytes(Header.StringsOffset);
  }

  std::unique_ptr<const MemoryBuffer> getBuffer() const {
    return MemoryBuffer::getMemBuffer(
        StringRef(reinterpret_cast<const char *>(this), sizeof(MapFile)),
        "header",
        /* RequresNullTerminator */ false);
  }
};

// The header map hash function.
static inline unsigned getHash(StringRef Str) {
  unsigned Result = 0;
  for (char C : Str)
    Result += toLowercase(C) * 13;
  return Result;
}

template <class FileTy> struct FileMaker {
  FileTy &File;
  unsigned SI = 1;
  unsigned BI = 0;
  FileMaker(FileTy &File) : File(File) {}

  unsigned addString(StringRef S) {
    assert(SI + S.size() + 1 <= sizeof(File.Bytes));
    std::copy(S.begin(), S.end(), File.Bytes + SI);
    auto OldSI = SI;
    SI += S.size() + 1;
    return OldSI;
  }
  void addBucket(unsigned Hash, unsigned Key, unsigned Prefix, unsigned Suffix) {
    assert(!(File.Header.NumBuckets & (File.Header.NumBuckets - 1)));
    unsigned I = Hash & (File.Header.NumBuckets - 1);
    do {
      if (!File.Buckets[I].Key) {
        File.Buckets[I].Key = Key;
        File.Buckets[I].Prefix = Prefix;
        File.Buckets[I].Suffix = Suffix;
        ++File.Header.NumEntries;
        return;
      }
      ++I;
      I &= File.Header.NumBuckets - 1;
    } while (I != (Hash & (File.Header.NumBuckets - 1)));
    llvm_unreachable("no empty buckets");
  }
};

TEST(HeaderMapTest, checkHeaderEmpty) {
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(
      *MemoryBuffer::getMemBufferCopy("", "empty"), NeedsSwap));
  ASSERT_FALSE(HeaderMapImpl::checkHeader(
      *MemoryBuffer::getMemBufferCopy("", "empty"), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderMagic) {
  MapFile<1, 1> File;
  File.init();
  File.Header.Magic = 0;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderReserved) {
  MapFile<1, 1> File;
  File.init();
  File.Header.Reserved = 1;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderVersion) {
  MapFile<1, 1> File;
  File.init();
  ++File.Header.Version;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderValidButEmpty) {
  MapFile<1, 1> File;
  File.init();
  bool NeedsSwap;
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_FALSE(NeedsSwap);

  File.swapBytes();
  ASSERT_TRUE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
  ASSERT_TRUE(NeedsSwap);
}

TEST(HeaderMapTest, checkHeader3Buckets) {
  MapFile<3, 1> File;
  ASSERT_EQ(3 * sizeof(HMapBucket), sizeof(File.Buckets));

  File.init();
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeader0Buckets) {
  // Create with 1 bucket to avoid 0-sized arrays.
  MapFile<1, 1> File;
  File.init();
  File.Header.NumBuckets = 0;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, checkHeaderNotEnoughBuckets) {
  MapFile<1, 1> File;
  File.init();
  File.Header.NumBuckets = 8;
  bool NeedsSwap;
  ASSERT_FALSE(HeaderMapImpl::checkHeader(*File.getBuffer(), NeedsSwap));
}

TEST(HeaderMapTest, lookupFilename) {
  typedef MapFile<2, 7> FileTy;
  FileTy File;
  File.init();

  FileMaker<FileTy> Maker(File);
  auto a = Maker.addString("a");
  auto b = Maker.addString("b");
  auto c = Maker.addString("c");
  Maker.addBucket(getHash("a"), a, b, c);

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
  typedef MapFile<2, 64 - sizeof(HMapHeader) - 2 * sizeof(HMapBucket)> FileTy;
  static_assert(std::is_standard_layout<FileTy>::value,
                "Expected standard layout");
  static_assert(sizeof(FileTy) == 64, "check the math");
  PaddedFile<FileTy, uint64_t> P;
  auto &File = P.File;
  auto &Padding = P.Padding;
  File.init();

  FileMaker<FileTy> Maker(File);
  auto a = Maker.addString("a");
  auto b = Maker.addString("b");
  auto c = Maker.addString("c");
  Maker.addBucket(getHash("a"), a, b, c);

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
  typedef MapFile<2, 64 - sizeof(HMapHeader) - 2 * sizeof(HMapBucket)> FileTy;
  static_assert(std::is_standard_layout<FileTy>::value,
                "Expected standard layout");
  static_assert(sizeof(FileTy) == 64, "check the math");
  PaddedFile<FileTy, uint64_t> P;
  auto &File = P.File;
  auto &Padding = P.Padding;
  File.init();

  FileMaker<FileTy> Maker(File);
  auto a = Maker.addString("a");
  auto c = Maker.addString("c");
  auto b = Maker.addString("b"); // Store the prefix last.
  Maker.addBucket(getHash("a"), a, b, c);

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
