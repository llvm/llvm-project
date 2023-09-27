//===- unittests/Lex/HeaderMapTestUtils.h - HeaderMap utils -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_LEX_HEADERMAPTESTUTILS_H
#define LLVM_CLANG_UNITTESTS_LEX_HEADERMAPTESTUTILS_H

#include "clang/Basic/CharInfo.h"
#include "clang/Lex/HeaderMap.h"
#include "clang/Lex/HeaderMapTypes.h"
#include "llvm/Support/SwapByteOrder.h"
#include <cassert>

namespace clang {
namespace test {

// Lay out a header file for testing.
template <unsigned NumBuckets, unsigned NumBytes> struct HMapFileMock {
  HMapHeader Header;
  HMapBucket Buckets[NumBuckets];
  unsigned char Bytes[NumBytes];

  void init() {
    memset(this, 0, sizeof(HMapFileMock));
    Header.Magic = HMAP_HeaderMagicNumber;
    Header.Version = HMAP_HeaderVersion;
    Header.NumBuckets = NumBuckets;
    Header.StringsOffset = sizeof(Header) + sizeof(Buckets);
  }

  void swapBytes() {
    Header.Magic = llvm::byteswap(Header.Magic);
    Header.Version = llvm::byteswap(Header.Version);
    Header.NumBuckets = llvm::byteswap(Header.NumBuckets);
    Header.StringsOffset = llvm::byteswap(Header.StringsOffset);
  }

  std::unique_ptr<llvm::MemoryBuffer> getBuffer() {
    return llvm::MemoryBuffer::getMemBuffer(
        StringRef(reinterpret_cast<char *>(this), sizeof(HMapFileMock)),
        "header",
        /* RequresNullTerminator */ false);
  }
};

template <class FileTy> struct HMapFileMockMaker {
  FileTy &File;
  unsigned SI = 1;
  unsigned BI = 0;
  HMapFileMockMaker(FileTy &File) : File(File) {}

  unsigned addString(StringRef S) {
    assert(SI + S.size() + 1 <= sizeof(File.Bytes));
    std::copy(S.begin(), S.end(), File.Bytes + SI);
    auto OldSI = SI;
    SI += S.size() + 1;
    return OldSI;
  }

  void addBucket(StringRef Str, unsigned Key, unsigned Prefix,
                 unsigned Suffix) {
    addBucket(getHash(Str), Key, Prefix, Suffix);
  }

  void addBucket(unsigned Hash, unsigned Key, unsigned Prefix,
                 unsigned Suffix) {
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

  // The header map hash function.
  static unsigned getHash(StringRef Str) {
    unsigned Result = 0;
    for (char C : Str)
      Result += toLowercase(C) * 13;
    return Result;
  }
};

} // namespace test
} // namespace clang

#endif
