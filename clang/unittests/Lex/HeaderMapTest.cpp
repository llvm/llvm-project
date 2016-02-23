//===- unittests/Lex/HeaderMapTest.cpp - HeaderMap tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------===//

#include "clang/Lex/HeaderMap.h"
#include "clang/Lex/HeaderMapTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SwapByteOrder.h"
#include "gtest/gtest.h"

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

} // end namespace
