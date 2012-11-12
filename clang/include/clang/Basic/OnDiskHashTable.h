//===--- OnDiskHashTable.h - On-Disk Hash Table Implementation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines facilities for reading and writing on-disk hash tables.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_ON_DISK_HASH_TABLE_H
#define LLVM_CLANG_BASIC_ON_DISK_HASH_TABLE_H

#include "llvm/Support/Allocator.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Host.h"
#include <cassert>
#include <cstdlib>

namespace clang {

namespace io {

typedef uint32_t Offset;

inline void Emit8(raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
}

inline void Emit16(raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  assert((V >> 16) == 0);
}

inline void Emit24(raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  Out << (unsigned char)(V >> 16);
  assert((V >> 24) == 0);
}

inline void Emit32(raw_ostream& Out, uint32_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  Out << (unsigned char)(V >> 16);
  Out << (unsigned char)(V >> 24);
}

inline void Emit64(raw_ostream& Out, uint64_t V) {
  Out << (unsigned char)(V);
  Out << (unsigned char)(V >>  8);
  Out << (unsigned char)(V >> 16);
  Out << (unsigned char)(V >> 24);
  Out << (unsigned char)(V >> 32);
  Out << (unsigned char)(V >> 40);
  Out << (unsigned char)(V >> 48);
  Out << (unsigned char)(V >> 56);
}

inline void Pad(raw_ostream& Out, unsigned A) {
  Offset off = (Offset) Out.tell();
  for (uint32_t n = llvm::OffsetToAlignment(off, A); n; --n)
    Emit8(Out, 0);
}

inline uint16_t ReadUnalignedLE16(const unsigned char *&Data) {
  uint16_t V = ((uint16_t)Data[0]) |
               ((uint16_t)Data[1] <<  8);
  Data += 2;
  return V;
}

inline uint32_t ReadUnalignedLE32(const unsigned char *&Data) {
  uint32_t V = ((uint32_t)Data[0])  |
               ((uint32_t)Data[1] << 8)  |
               ((uint32_t)Data[2] << 16) |
               ((uint32_t)Data[3] << 24);
  Data += 4;
  return V;
}

inline uint64_t ReadUnalignedLE64(const unsigned char *&Data) {
  uint64_t V = ((uint64_t)Data[0])  |
    ((uint64_t)Data[1] << 8)  |
    ((uint64_t)Data[2] << 16) |
    ((uint64_t)Data[3] << 24) |
    ((uint64_t)Data[4] << 32) |
    ((uint64_t)Data[5] << 40) |
    ((uint64_t)Data[6] << 48) |
    ((uint64_t)Data[7] << 56);
  Data += 8;
  return V;
}

inline uint32_t ReadLE32(const unsigned char *&Data) {
  // Hosts that directly support little-endian 32-bit loads can just
  // use them.  Big-endian hosts need a bswap.
  uint32_t V = *((const uint32_t*)Data);
  if (llvm::sys::isBigEndianHost())
    V = llvm::ByteSwap_32(V);
  Data += 4;
  return V;
}

} // end namespace io

template<typename Info>
class OnDiskChainedHashTableGenerator {
  unsigned NumBuckets;
  unsigned NumEntries;
  llvm::BumpPtrAllocator BA;

  class Item {
  public:
    typename Info::key_type key;
    typename Info::data_type data;
    Item *next;
    const uint32_t hash;

    Item(typename Info::key_type_ref k, typename Info::data_type_ref d,
         Info &InfoObj)
    : key(k), data(d), next(0), hash(InfoObj.ComputeHash(k)) {}
  };

  class Bucket {
  public:
    io::Offset off;
    Item* head;
    unsigned length;

    Bucket() {}
  };

  Bucket* Buckets;

private:
  void insert(Bucket* b, size_t size, Item* E) {
    unsigned idx = E->hash & (size - 1);
    Bucket& B = b[idx];
    E->next = B.head;
    ++B.length;
    B.head = E;
  }

  void resize(size_t newsize) {
    Bucket* newBuckets = (Bucket*) std::calloc(newsize, sizeof(Bucket));
    // Populate newBuckets with the old entries.
    for (unsigned i = 0; i < NumBuckets; ++i)
      for (Item* E = Buckets[i].head; E ; ) {
        Item* N = E->next;
        E->next = 0;
        insert(newBuckets, newsize, E);
        E = N;
      }

    free(Buckets);
    NumBuckets = newsize;
    Buckets = newBuckets;
  }

public:

  void insert(typename Info::key_type_ref key,
              typename Info::data_type_ref data) {
    Info InfoObj;
    insert(key, data, InfoObj);
  }

  void insert(typename Info::key_type_ref key,
              typename Info::data_type_ref data, Info &InfoObj) {

    ++NumEntries;
    if (4*NumEntries >= 3*NumBuckets) resize(NumBuckets*2);
    insert(Buckets, NumBuckets, new (BA.Allocate<Item>()) Item(key, data,
                                                               InfoObj));
  }

  io::Offset Emit(raw_ostream &out) {
    Info InfoObj;
    return Emit(out, InfoObj);
  }

  io::Offset Emit(raw_ostream &out, Info &InfoObj) {
    using namespace clang::io;

    // Emit the payload of the table.
    for (unsigned i = 0; i < NumBuckets; ++i) {
      Bucket& B = Buckets[i];
      if (!B.head) continue;

      // Store the offset for the data of this bucket.
      B.off = out.tell();
      assert(B.off && "Cannot write a bucket at offset 0. Please add padding.");

      // Write out the number of items in the bucket.
      Emit16(out, B.length);
      assert(B.length != 0  && "Bucket has a head but zero length?");

      // Write out the entries in the bucket.
      for (Item *I = B.head; I ; I = I->next) {
        Emit32(out, I->hash);
        const std::pair<unsigned, unsigned>& Len =
          InfoObj.EmitKeyDataLength(out, I->key, I->data);
        InfoObj.EmitKey(out, I->key, Len.first);
        InfoObj.EmitData(out, I->key, I->data, Len.second);
      }
    }

    // Emit the hashtable itself.
    Pad(out, 4);
    io::Offset TableOff = out.tell();
    Emit32(out, NumBuckets);
    Emit32(out, NumEntries);
    for (unsigned i = 0; i < NumBuckets; ++i) Emit32(out, Buckets[i].off);

    return TableOff;
  }

  OnDiskChainedHashTableGenerator() {
    NumEntries = 0;
    NumBuckets = 64;
    // Note that we do not need to run the constructors of the individual
    // Bucket objects since 'calloc' returns bytes that are all 0.
    Buckets = (Bucket*) std::calloc(NumBuckets, sizeof(Bucket));
  }

  ~OnDiskChainedHashTableGenerator() {
    std::free(Buckets);
  }
};

template<typename Info>
class OnDiskChainedHashTable {
  const unsigned NumBuckets;
  const unsigned NumEntries;
  const unsigned char* const Buckets;
  const unsigned char* const Base;
  Info InfoObj;

public:
  typedef typename Info::internal_key_type internal_key_type;
  typedef typename Info::external_key_type external_key_type;
  typedef typename Info::data_type         data_type;

  OnDiskChainedHashTable(unsigned numBuckets, unsigned numEntries,
                         const unsigned char* buckets,
                         const unsigned char* base,
                         const Info &InfoObj = Info())
    : NumBuckets(numBuckets), NumEntries(numEntries),
      Buckets(buckets), Base(base), InfoObj(InfoObj) {
        assert((reinterpret_cast<uintptr_t>(buckets) & 0x3) == 0 &&
               "'buckets' must have a 4-byte alignment");
      }

  unsigned getNumBuckets() const { return NumBuckets; }
  unsigned getNumEntries() const { return NumEntries; }
  const unsigned char* getBase() const { return Base; }
  const unsigned char* getBuckets() const { return Buckets; }

  bool isEmpty() const { return NumEntries == 0; }

  class iterator {
    internal_key_type key;
    const unsigned char* const data;
    const unsigned len;
    Info *InfoObj;
  public:
    iterator() : data(0), len(0) {}
    iterator(const internal_key_type k, const unsigned char* d, unsigned l,
             Info *InfoObj)
      : key(k), data(d), len(l), InfoObj(InfoObj) {}

    data_type operator*() const { return InfoObj->ReadData(key, data, len); }
    bool operator==(const iterator& X) const { return X.data == data; }
    bool operator!=(const iterator& X) const { return X.data != data; }
  };

  iterator find(const external_key_type& eKey, Info *InfoPtr = 0) {
    if (!InfoPtr)
      InfoPtr = &InfoObj;

    using namespace io;
    const internal_key_type& iKey = InfoObj.GetInternalKey(eKey);
    unsigned key_hash = InfoObj.ComputeHash(iKey);

    // Each bucket is just a 32-bit offset into the hash table file.
    unsigned idx = key_hash & (NumBuckets - 1);
    const unsigned char* Bucket = Buckets + sizeof(uint32_t)*idx;

    unsigned offset = ReadLE32(Bucket);
    if (offset == 0) return iterator(); // Empty bucket.
    const unsigned char* Items = Base + offset;

    // 'Items' starts with a 16-bit unsigned integer representing the
    // number of items in this bucket.
    unsigned len = ReadUnalignedLE16(Items);

    for (unsigned i = 0; i < len; ++i) {
      // Read the hash.
      uint32_t item_hash = ReadUnalignedLE32(Items);

      // Determine the length of the key and the data.
      const std::pair<unsigned, unsigned>& L = Info::ReadKeyDataLength(Items);
      unsigned item_len = L.first + L.second;

      // Compare the hashes.  If they are not the same, skip the entry entirely.
      if (item_hash != key_hash) {
        Items += item_len;
        continue;
      }

      // Read the key.
      const internal_key_type& X =
        InfoPtr->ReadKey((const unsigned char* const) Items, L.first);

      // If the key doesn't match just skip reading the value.
      if (!InfoPtr->EqualKey(X, iKey)) {
        Items += item_len;
        continue;
      }

      // The key matches!
      return iterator(X, Items + L.first, L.second, InfoPtr);
    }

    return iterator();
  }

  iterator end() const { return iterator(); }

  /// \brief Iterates over all of the keys in the table.
  class key_iterator {
    const unsigned char* Ptr;
    unsigned NumItemsInBucketLeft;
    unsigned NumEntriesLeft;
    Info *InfoObj;
  public:
    typedef external_key_type value_type;

    key_iterator(const unsigned char* const Ptr, unsigned NumEntries,
                  Info *InfoObj)
      : Ptr(Ptr), NumItemsInBucketLeft(0), NumEntriesLeft(NumEntries),
        InfoObj(InfoObj) { }
    key_iterator()
      : Ptr(0), NumItemsInBucketLeft(0), NumEntriesLeft(0), InfoObj(0) { }

    friend bool operator==(const key_iterator &X, const key_iterator &Y) {
      return X.NumEntriesLeft == Y.NumEntriesLeft;
    }
    friend bool operator!=(const key_iterator& X, const key_iterator &Y) {
      return X.NumEntriesLeft != Y.NumEntriesLeft;
    }

    key_iterator& operator++() {  // Preincrement
      if (!NumItemsInBucketLeft) {
        // 'Items' starts with a 16-bit unsigned integer representing the
        // number of items in this bucket.
        NumItemsInBucketLeft = io::ReadUnalignedLE16(Ptr);
      }
      Ptr += 4; // Skip the hash.
      // Determine the length of the key and the data.
      const std::pair<unsigned, unsigned>& L = Info::ReadKeyDataLength(Ptr);
      Ptr += L.first + L.second;
      assert(NumItemsInBucketLeft);
      --NumItemsInBucketLeft;
      assert(NumEntriesLeft);
      --NumEntriesLeft;
      return *this;
    }
    key_iterator operator++(int) {  // Postincrement
      key_iterator tmp = *this; ++*this; return tmp;
    }

    value_type operator*() const {
      const unsigned char* LocalPtr = Ptr;
      if (!NumItemsInBucketLeft)
        LocalPtr += 2; // number of items in bucket
      LocalPtr += 4; // Skip the hash.

      // Determine the length of the key and the data.
      const std::pair<unsigned, unsigned>& L
        = Info::ReadKeyDataLength(LocalPtr);

      // Read the key.
      const internal_key_type& Key = InfoObj->ReadKey(LocalPtr, L.first);
      return InfoObj->GetExternalKey(Key);
    }
  };

  key_iterator key_begin() {
    return key_iterator(Base + 4, getNumEntries(), &InfoObj);
  }
  key_iterator key_end() { return key_iterator(); }

  /// \brief Iterates over all the entries in the table, returning the data.
  class data_iterator {
    const unsigned char* Ptr;
    unsigned NumItemsInBucketLeft;
    unsigned NumEntriesLeft;
    Info *InfoObj;
  public:
    typedef data_type value_type;

    data_iterator(const unsigned char* const Ptr, unsigned NumEntries,
                  Info *InfoObj)
      : Ptr(Ptr), NumItemsInBucketLeft(0), NumEntriesLeft(NumEntries),
        InfoObj(InfoObj) { }
    data_iterator()
      : Ptr(0), NumItemsInBucketLeft(0), NumEntriesLeft(0), InfoObj(0) { }

    bool operator==(const data_iterator& X) const {
      return X.NumEntriesLeft == NumEntriesLeft;
    }
    bool operator!=(const data_iterator& X) const {
      return X.NumEntriesLeft != NumEntriesLeft;
    }

    data_iterator& operator++() {  // Preincrement
      if (!NumItemsInBucketLeft) {
        // 'Items' starts with a 16-bit unsigned integer representing the
        // number of items in this bucket.
        NumItemsInBucketLeft = io::ReadUnalignedLE16(Ptr);
      }
      Ptr += 4; // Skip the hash.
      // Determine the length of the key and the data.
      const std::pair<unsigned, unsigned>& L = Info::ReadKeyDataLength(Ptr);
      Ptr += L.first + L.second;
      assert(NumItemsInBucketLeft);
      --NumItemsInBucketLeft;
      assert(NumEntriesLeft);
      --NumEntriesLeft;
      return *this;
    }
    data_iterator operator++(int) {  // Postincrement
      data_iterator tmp = *this; ++*this; return tmp;
    }

    value_type operator*() const {
      const unsigned char* LocalPtr = Ptr;
      if (!NumItemsInBucketLeft)
        LocalPtr += 2; // number of items in bucket
      LocalPtr += 4; // Skip the hash.

      // Determine the length of the key and the data.
      const std::pair<unsigned, unsigned>& L =Info::ReadKeyDataLength(LocalPtr);

      // Read the key.
      const internal_key_type& Key =
        InfoObj->ReadKey(LocalPtr, L.first);
      return InfoObj->ReadData(Key, LocalPtr + L.first, L.second);
    }
  };

  data_iterator data_begin() {
    return data_iterator(Base + 4, getNumEntries(), &InfoObj);
  }
  data_iterator data_end() { return data_iterator(); }

  Info &getInfoObj() { return InfoObj; }

  static OnDiskChainedHashTable* Create(const unsigned char* buckets,
                                        const unsigned char* const base,
                                        const Info &InfoObj = Info()) {
    using namespace io;
    assert(buckets > base);
    assert((reinterpret_cast<uintptr_t>(buckets) & 0x3) == 0 &&
           "buckets should be 4-byte aligned.");

    unsigned numBuckets = ReadLE32(buckets);
    unsigned numEntries = ReadLE32(buckets);
    return new OnDiskChainedHashTable<Info>(numBuckets, numEntries, buckets,
                                            base, InfoObj);
  }
};

} // end namespace clang

#endif
