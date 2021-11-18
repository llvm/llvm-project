//===- sanitizer_dense_map.h - Dense probed hash table ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is fork of llvm/ADT/DenseMap.h class.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_DENSE_MAP_H
#define SANITIZER_DENSE_MAP_H

#include "sanitizer_common.h"
#include "sanitizer_dense_map_info.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {
namespace detail {

// We extend a pair to allow users to override the bucket type with their own
// implementation without requiring two members.
template <typename KeyT, typename ValueT>
struct DenseMapPair : public std::pair<KeyT, ValueT> {
  using std::pair<KeyT, ValueT>::pair;

  KeyT &getFirst() { return std::pair<KeyT, ValueT>::first; }
  const KeyT &getFirst() const { return std::pair<KeyT, ValueT>::first; }
  ValueT &getSecond() { return std::pair<KeyT, ValueT>::second; }
  const ValueT &getSecond() const { return std::pair<KeyT, ValueT>::second; }
};

}  // end namespace detail

template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename Bucket = llvm::detail::DenseMapPair<KeyT, ValueT>,
          bool IsConst = false>
class DenseMapIterator;

template <typename DerivedT, typename KeyT, typename ValueT, typename KeyInfoT,
          typename BucketT>
class DenseMapBase : public DebugEpochBase {
  template <typename T>
  using const_arg_type_t = typename const_pointer_or_const_ref<T>::type;

 public:
  using size_type = unsigned;
  using key_type = KeyT;
  using mapped_type = ValueT;
  using value_type = BucketT;

  LLVM_NODISCARD bool empty() const { return getNumEntries() == 0; }
  unsigned size() const { return getNumEntries(); }

  /// Grow the densemap so that it can contain at least \p NumEntries items
  /// before resizing again.
  void reserve(size_type NumEntries) {
    auto NumBuckets = getMinBucketToReserveForEntries(NumEntries);
    incrementEpoch();
    if (NumBuckets > getNumBuckets())
      grow(NumBuckets);
  }

  void clear() {
    incrementEpoch();
    if (getNumEntries() == 0 && getNumTombstones() == 0)
      return;

    // If the capacity of the array is huge, and the # elements used is small,
    // shrink the array.
    if (getNumEntries() * 4 < getNumBuckets() && getNumBuckets() > 64) {
      shrink_and_clear();
      return;
    }

    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    if (std::is_trivially_destructible<ValueT>::value) {
      // Use a simpler loop when values don't need destruction.
      for (BucketT *P = getBuckets(), *E = getBucketsEnd(); P != E; ++P)
        P->getFirst() = EmptyKey;
    } else {
      unsigned NumEntries = getNumEntries();
      for (BucketT *P = getBuckets(), *E = getBucketsEnd(); P != E; ++P) {
        if (!KeyInfoT::isEqual(P->getFirst(), EmptyKey)) {
          if (!KeyInfoT::isEqual(P->getFirst(), TombstoneKey)) {
            P->getSecond().~ValueT();
            --NumEntries;
          }
          P->getFirst() = EmptyKey;
        }
      }
      assert(NumEntries == 0 && "Node count imbalance!");
    }
    setNumEntries(0);
    setNumTombstones(0);
  }

  /// Return 1 if the specified key is in the map, 0 otherwise.
  size_type count(const_arg_type_t<KeyT> Val) const {
    const BucketT *TheBucket;
    return LookupBucketFor(Val, TheBucket) ? 1 : 0;
  }

  iterator find(const_arg_type_t<KeyT> Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return makeIterator(
          TheBucket,
          shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(), *this,
          true);
    return end();
  }
  const_iterator find(const_arg_type_t<KeyT> Val) const {
    const BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return makeConstIterator(
          TheBucket,
          shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(), *this,
          true);
    return end();
  }

  /// Alternate version of find() which allows a different, and possibly
  /// less expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key
  /// type used.
  template <class LookupKeyT>
  iterator find_as(const LookupKeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return makeIterator(
          TheBucket,
          shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(), *this,
          true);
    return end();
  }
  template <class LookupKeyT>
  const_iterator find_as(const LookupKeyT &Val) const {
    const BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return makeConstIterator(
          TheBucket,
          shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(), *this,
          true);
    return end();
  }

  /// lookup - Return the entry for the specified key, or a default
  /// constructed value if no such entry exists.
  ValueT lookup(const_arg_type_t<KeyT> Val) const {
    const BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return TheBucket->getSecond();
    return ValueT();
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // If the key is already in the map, it returns false and doesn't update the
  // value.
  std::pair<iterator, bool> insert(const std::pair<KeyT, ValueT> &KV) {
    return try_emplace(KV.first, KV.second);
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // If the key is already in the map, it returns false and doesn't update the
  // value.
  std::pair<iterator, bool> insert(std::pair<KeyT, ValueT> &&KV) {
    return try_emplace(std::move(KV.first), std::move(KV.second));
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // The value is constructed in-place if the key is not in the map, otherwise
  // it is not moved.
  template <typename... Ts>
  std::pair<iterator, bool> try_emplace(KeyT &&Key, Ts &&...Args) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return std::make_pair(
          makeIterator(
              TheBucket,
              shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(),
              *this, true),
          false);  // Already in map.

    // Otherwise, insert the new element.
    TheBucket =
        InsertIntoBucket(TheBucket, std::move(Key), std::forward<Ts>(Args)...);
    return std::make_pair(
        makeIterator(
            TheBucket,
            shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(),
            *this, true),
        true);
  }

  // Inserts key,value pair into the map if the key isn't already in the map.
  // The value is constructed in-place if the key is not in the map, otherwise
  // it is not moved.
  template <typename... Ts>
  std::pair<iterator, bool> try_emplace(const KeyT &Key, Ts &&...Args) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return std::make_pair(
          makeIterator(
              TheBucket,
              shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(),
              *this, true),
          false);  // Already in map.

    // Otherwise, insert the new element.
    TheBucket = InsertIntoBucket(TheBucket, Key, std::forward<Ts>(Args)...);
    return std::make_pair(
        makeIterator(
            TheBucket,
            shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(),
            *this, true),
        true);
  }

  /// Alternate version of insert() which allows a different, and possibly
  /// less expensive, key type.
  /// The DenseMapInfo is responsible for supplying methods
  /// getHashValue(LookupKeyT) and isEqual(LookupKeyT, KeyT) for each key
  /// type used.
  template <typename LookupKeyT>
  std::pair<iterator, bool> insert_as(std::pair<KeyT, ValueT> &&KV,
                                      const LookupKeyT &Val) {
    BucketT *TheBucket;
    if (LookupBucketFor(Val, TheBucket))
      return std::make_pair(
          makeIterator(
              TheBucket,
              shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(),
              *this, true),
          false);  // Already in map.

    // Otherwise, insert the new element.
    TheBucket = InsertIntoBucketWithLookup(TheBucket, std::move(KV.first),
                                           std::move(KV.second), Val);
    return std::make_pair(
        makeIterator(
            TheBucket,
            shouldReverseIterate<KeyT>() ? getBuckets() : getBucketsEnd(),
            *this, true),
        true);
  }

  /// insert - Range insertion of pairs.
  template <typename InputIt>
  void insert(InputIt I, InputIt E) {
    for (; I != E; ++I) insert(*I);
  }

  bool erase(const KeyT &Val) {
    BucketT *TheBucket;
    if (!LookupBucketFor(Val, TheBucket))
      return false;  // not in map.

    TheBucket->getSecond().~ValueT();
    TheBucket->getFirst() = getTombstoneKey();
    decrementNumEntries();
    incrementNumTombstones();
    return true;
  }
  void erase(iterator I) {
    BucketT *TheBucket = &*I;
    TheBucket->getSecond().~ValueT();
    TheBucket->getFirst() = getTombstoneKey();
    decrementNumEntries();
    incrementNumTombstones();
  }

  value_type &FindAndConstruct(const KeyT &Key) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return *TheBucket;

    return *InsertIntoBucket(TheBucket, Key);
  }

  ValueT &operator[](const KeyT &Key) { return FindAndConstruct(Key).second; }

  value_type &FindAndConstruct(KeyT &&Key) {
    BucketT *TheBucket;
    if (LookupBucketFor(Key, TheBucket))
      return *TheBucket;

    return *InsertIntoBucket(TheBucket, std::move(Key));
  }

  ValueT &operator[](KeyT &&Key) {
    return FindAndConstruct(std::move(Key)).second;
  }

 protected:
  DenseMapBase() = default;

  void destroyAll() {
    if (getNumBuckets() == 0)  // Nothing to do.
      return;

    const KeyT EmptyKey = getEmptyKey(), TombstoneKey = getTombstoneKey();
    for (BucketT *P = getBuckets(), *E = getBucketsEnd(); P != E; ++P) {
      if (!KeyInfoT::isEqual(P->getFirst(), EmptyKey) &&
          !KeyInfoT::isEqual(P->getFirst(), TombstoneKey))
        P->getSecond().~ValueT();
      P->getFirst().~KeyT();
    }
  }

  void initEmpty() {
    setNumEntries(0);
    setNumTombstones(0);

    assert((getNumBuckets() & (getNumBuckets() - 1)) == 0 &&
           "# initial buckets must be a power of two!");
    const KeyT EmptyKey = getEmptyKey();
    for (BucketT *B = getBuckets(), *E = getBucketsEnd(); B != E; ++B)
      ::new (&B->getFirst()) KeyT(EmptyKey);
  }

  /// Returns the number of buckets to allocate to ensure that the DenseMap can
  /// accommodate \p NumEntries without need to grow().
  unsigned getMinBucketToReserveForEntries(unsigned NumEntries) {
    // Ensure that "NumEntries * 4 < NumBuckets * 3"
    if (NumEntries == 0)
      return 0;
    // +1 is required because of the strict equality.
    // For example if NumEntries is 48, we need to return 401.
    return NextPowerOf2(NumEntries * 4 / 3 + 1);
  }

  void moveFromOldBuckets(BucketT *OldBucketsBegin, BucketT *OldBucketsEnd) {
    initEmpty();

    // Insert all the old elements.
    const KeyT EmptyKey = getEmptyKey();
    const KeyT TombstoneKey = getTombstoneKey();
    for (BucketT *B = OldBucketsBegin, *E = OldBucketsEnd; B != E; ++B) {
      if (!KeyInfoT::isEqual(B->getFirst(), EmptyKey) &&
          !KeyInfoT::isEqual(B->getFirst(), TombstoneKey)) {
        // Insert the key/value into the new table.
        BucketT *DestBucket;
        bool FoundVal = LookupBucketFor(B->getFirst(), DestBucket);
        (void)FoundVal;  // silence warning.
        assert(!FoundVal && "Key already in new map?");
        DestBucket->getFirst() = std::move(B->getFirst());
        ::new (&DestBucket->getSecond()) ValueT(std::move(B->getSecond()));
        incrementNumEntries();

        // Free the value.
        B->getSecond().~ValueT();
      }
      B->getFirst().~KeyT();
    }
  }

  template <typename OtherBaseT>
  void copyFrom(
      const DenseMapBase<OtherBaseT, KeyT, ValueT, KeyInfoT, BucketT> &other) {
    assert(&other != this);
    assert(getNumBuckets() == other.getNumBuckets());

    setNumEntries(other.getNumEntries());
    setNumTombstones(other.getNumTombstones());

    if (std::is_trivially_copyable<KeyT>::value &&
        std::is_trivially_copyable<ValueT>::value)
      memcpy(reinterpret_cast<void *>(getBuckets()), other.getBuckets(),
             getNumBuckets() * sizeof(BucketT));
    else
      for (size_t i = 0; i < getNumBuckets(); ++i) {
        ::new (&getBuckets()[i].getFirst())
            KeyT(other.getBuckets()[i].getFirst());
        if (!KeyInfoT::isEqual(getBuckets()[i].getFirst(), getEmptyKey()) &&
            !KeyInfoT::isEqual(getBuckets()[i].getFirst(), getTombstoneKey()))
          ::new (&getBuckets()[i].getSecond())
              ValueT(other.getBuckets()[i].getSecond());
      }
  }

  static unsigned getHashValue(const KeyT &Val) {
    return KeyInfoT::getHashValue(Val);
  }

  template <typename LookupKeyT>
  static unsigned getHashValue(const LookupKeyT &Val) {
    return KeyInfoT::getHashValue(Val);
  }

  static const KeyT getEmptyKey() {
    static_assert(std::is_base_of<DenseMapBase, DerivedT>::value,
                  "Must pass the derived type to this template!");
    return KeyInfoT::getEmptyKey();
  }

  static const KeyT getTombstoneKey() { return KeyInfoT::getTombstoneKey(); }

 private:
  unsigned getNumEntries() const {
    return static_cast<const DerivedT *>(this)->getNumEntries();
  }

  void setNumEntries(unsigned Num) {
    static_cast<DerivedT *>(this)->setNumEntries(Num);
  }

  void incrementNumEntries() { setNumEntries(getNumEntries() + 1); }

  void decrementNumEntries() { setNumEntries(getNumEntries() - 1); }

  unsigned getNumTombstones() const {
    return static_cast<const DerivedT *>(this)->getNumTombstones();
  }

  void setNumTombstones(unsigned Num) {
    static_cast<DerivedT *>(this)->setNumTombstones(Num);
  }

  void incrementNumTombstones() { setNumTombstones(getNumTombstones() + 1); }

  void decrementNumTombstones() { setNumTombstones(getNumTombstones() - 1); }

  const BucketT *getBuckets() const {
    return static_cast<const DerivedT *>(this)->getBuckets();
  }

  BucketT *getBuckets() { return static_cast<DerivedT *>(this)->getBuckets(); }

  unsigned getNumBuckets() const {
    return static_cast<const DerivedT *>(this)->getNumBuckets();
  }

  BucketT *getBucketsEnd() { return getBuckets() + getNumBuckets(); }

  const BucketT *getBucketsEnd() const {
    return getBuckets() + getNumBuckets();
  }

  void grow(unsigned AtLeast) { static_cast<DerivedT *>(this)->grow(AtLeast); }

  template <typename KeyArg, typename... ValueArgs>
  BucketT *InsertIntoBucket(BucketT *TheBucket, KeyArg &&Key,
                            ValueArgs &&...Values) {
    TheBucket = InsertIntoBucketImpl(Key, Key, TheBucket);

    TheBucket->getFirst() = std::forward<KeyArg>(Key);
    ::new (&TheBucket->getSecond()) ValueT(std::forward<ValueArgs>(Values)...);
    return TheBucket;
  }

  template <typename LookupKeyT>
  BucketT *InsertIntoBucketWithLookup(BucketT *TheBucket, KeyT &&Key,
                                      ValueT &&Value, LookupKeyT &Lookup) {
    TheBucket = InsertIntoBucketImpl(Key, Lookup, TheBucket);

    TheBucket->getFirst() = std::move(Key);
    ::new (&TheBucket->getSecond()) ValueT(std::move(Value));
    return TheBucket;
  }

  template <typename LookupKeyT>
  BucketT *InsertIntoBucketImpl(const KeyT &Key, const LookupKeyT &Lookup,
                                BucketT *TheBucket) {
    // If the load of the hash table is more than 3/4, or if fewer than 1/8 of
    // the buckets are empty (meaning that many are filled with tombstones),
    // grow the table.
    //
    // The later case is tricky.  For example, if we had one empty bucket with
    // tons of tombstones, failing lookups (e.g. for insertion) would have to
    // probe almost the entire table until it found the empty bucket.  If the
    // table completely filled with tombstones, no lookup would ever succeed,
    // causing infinite loops in lookup.
    unsigned NewNumEntries = getNumEntries() + 1;
    unsigned NumBuckets = getNumBuckets();
    if (LLVM_UNLIKELY(NewNumEntries * 4 >= NumBuckets * 3)) {
      this->grow(NumBuckets * 2);
      LookupBucketFor(Lookup, TheBucket);
      NumBuckets = getNumBuckets();
    } else if (LLVM_UNLIKELY(NumBuckets -
                                 (NewNumEntries + getNumTombstones()) <=
                             NumBuckets / 8)) {
      this->grow(NumBuckets);
      LookupBucketFor(Lookup, TheBucket);
    }
    assert(TheBucket);

    // Only update the state after we've grown our bucket space appropriately
    // so that when growing buckets we have self-consistent entry count.
    incrementNumEntries();

    // If we are writing over a tombstone, remember this.
    const KeyT EmptyKey = getEmptyKey();
    if (!KeyInfoT::isEqual(TheBucket->getFirst(), EmptyKey))
      decrementNumTombstones();

    return TheBucket;
  }

  /// LookupBucketFor - Lookup the appropriate bucket for Val, returning it in
  /// FoundBucket.  If the bucket contains the key and a value, this returns
  /// true, otherwise it returns a bucket with an empty marker or tombstone and
  /// returns false.
  template <typename LookupKeyT>
  bool LookupBucketFor(const LookupKeyT &Val,
                       const BucketT *&FoundBucket) const {
    const BucketT *BucketsPtr = getBuckets();
    const unsigned NumBuckets = getNumBuckets();

    if (NumBuckets == 0) {
      FoundBucket = nullptr;
      return false;
    }

    // FoundTombstone - Keep track of whether we find a tombstone while probing.
    const BucketT *FoundTombstone = nullptr;
    const KeyT EmptyKey = getEmptyKey();
    const KeyT TombstoneKey = getTombstoneKey();
    assert(!KeyInfoT::isEqual(Val, EmptyKey) &&
           !KeyInfoT::isEqual(Val, TombstoneKey) &&
           "Empty/Tombstone value shouldn't be inserted into map!");

    unsigned BucketNo = getHashValue(Val) & (NumBuckets - 1);
    unsigned ProbeAmt = 1;
    while (true) {
      const BucketT *ThisBucket = BucketsPtr + BucketNo;
      // Found Val's bucket?  If so, return it.
      if (LLVM_LIKELY(KeyInfoT::isEqual(Val, ThisBucket->getFirst()))) {
        FoundBucket = ThisBucket;
        return true;
      }

      // If we found an empty bucket, the key doesn't exist in the set.
      // Insert it and return the default value.
      if (LLVM_LIKELY(KeyInfoT::isEqual(ThisBucket->getFirst(), EmptyKey))) {
        // If we've already seen a tombstone while probing, fill it in instead
        // of the empty bucket we eventually probed to.
        FoundBucket = FoundTombstone ? FoundTombstone : ThisBucket;
        return false;
      }

      // If this is a tombstone, remember it.  If Val ends up not in the map, we
      // prefer to return it than something that would require more probing.
      if (KeyInfoT::isEqual(ThisBucket->getFirst(), TombstoneKey) &&
          !FoundTombstone)
        FoundTombstone = ThisBucket;  // Remember the first tombstone found.

      // Otherwise, it's a hash collision or a tombstone, continue quadratic
      // probing.
      BucketNo += ProbeAmt++;
      BucketNo &= (NumBuckets - 1);
    }
  }

  template <typename LookupKeyT>
  bool LookupBucketFor(const LookupKeyT &Val, BucketT *&FoundBucket) {
    const BucketT *ConstFoundBucket;
    bool Result = const_cast<const DenseMapBase *>(this)->LookupBucketFor(
        Val, ConstFoundBucket);
    FoundBucket = const_cast<BucketT *>(ConstFoundBucket);
    return Result;
  }

 public:
  /// Return the approximate size (in bytes) of the actual map.
  /// This is just the raw memory used by DenseMap.
  /// If entries are pointers to objects, the size of the referenced objects
  /// are not included.
  size_t getMemorySize() const { return getNumBuckets() * sizeof(BucketT); }
};

/// Inequality comparison for DenseMap.
///
/// Equivalent to !(LHS == RHS). See operator== for performance notes.
template <typename DerivedT, typename KeyT, typename ValueT, typename KeyInfoT,
          typename BucketT>
bool operator!=(
    const DenseMapBase<DerivedT, KeyT, ValueT, KeyInfoT, BucketT> &LHS,
    const DenseMapBase<DerivedT, KeyT, ValueT, KeyInfoT, BucketT> &RHS) {
  return !(LHS == RHS);
}

template <typename KeyT, typename ValueT,
          typename KeyInfoT = DenseMapInfo<KeyT>,
          typename BucketT = llvm::detail::DenseMapPair<KeyT, ValueT>>
class DenseMap : public DenseMapBase<DenseMap<KeyT, ValueT, KeyInfoT, BucketT>,
                                     KeyT, ValueT, KeyInfoT, BucketT> {
  friend class DenseMapBase<DenseMap, KeyT, ValueT, KeyInfoT, BucketT>;

  // Lift some types from the dependent base class into this class for
  // simplicity of referring to them.
  using BaseT = DenseMapBase<DenseMap, KeyT, ValueT, KeyInfoT, BucketT>;

  BucketT *Buckets;
  unsigned NumEntries;
  unsigned NumTombstones;
  unsigned NumBuckets;

 public:
  /// Create a DenseMap with an optional \p InitialReserve that guarantee that
  /// this number of elements can be inserted in the map without grow()
  explicit DenseMap(unsigned InitialReserve = 0) { init(InitialReserve); }

  DenseMap(const DenseMap &other) : BaseT() {
    init(0);
    copyFrom(other);
  }

  DenseMap(DenseMap &&other) : BaseT() {
    init(0);
    swap(other);
  }

  template <typename InputIt>
  DenseMap(const InputIt &I, const InputIt &E) {
    init(std::distance(I, E));
    this->insert(I, E);
  }

  DenseMap(std::initializer_list<typename BaseT::value_type> Vals) {
    init(Vals.size());
    this->insert(Vals.begin(), Vals.end());
  }

  ~DenseMap() {
    this->destroyAll();
    deallocate_buffer(Buckets, sizeof(BucketT) * NumBuckets, alignof(BucketT));
  }

  void swap(DenseMap &RHS) {
    this->incrementEpoch();
    RHS.incrementEpoch();
    std::swap(Buckets, RHS.Buckets);
    std::swap(NumEntries, RHS.NumEntries);
    std::swap(NumTombstones, RHS.NumTombstones);
    std::swap(NumBuckets, RHS.NumBuckets);
  }

  DenseMap &operator=(const DenseMap &other) {
    if (&other != this)
      copyFrom(other);
    return *this;
  }

  DenseMap &operator=(DenseMap &&other) {
    this->destroyAll();
    deallocate_buffer(Buckets, sizeof(BucketT) * NumBuckets, alignof(BucketT));
    init(0);
    swap(other);
    return *this;
  }

  void copyFrom(const DenseMap &other) {
    this->destroyAll();
    deallocate_buffer(Buckets, sizeof(BucketT) * NumBuckets, alignof(BucketT));
    if (allocateBuckets(other.NumBuckets)) {
      this->BaseT::copyFrom(other);
    } else {
      NumEntries = 0;
      NumTombstones = 0;
    }
  }

  void init(unsigned InitNumEntries) {
    auto InitBuckets = BaseT::getMinBucketToReserveForEntries(InitNumEntries);
    if (allocateBuckets(InitBuckets)) {
      this->BaseT::initEmpty();
    } else {
      NumEntries = 0;
      NumTombstones = 0;
    }
  }

  void grow(unsigned AtLeast) {
    unsigned OldNumBuckets = NumBuckets;
    BucketT *OldBuckets = Buckets;

    allocateBuckets(std::max<unsigned>(
        64, static_cast<unsigned>(NextPowerOf2(AtLeast - 1))));
    assert(Buckets);
    if (!OldBuckets) {
      this->BaseT::initEmpty();
      return;
    }

    this->moveFromOldBuckets(OldBuckets, OldBuckets + OldNumBuckets);

    // Free the old table.
    deallocate_buffer(OldBuckets, sizeof(BucketT) * OldNumBuckets,
                      alignof(BucketT));
  }

 private:
  unsigned getNumEntries() const { return NumEntries; }

  void setNumEntries(unsigned Num) { NumEntries = Num; }

  unsigned getNumTombstones() const { return NumTombstones; }

  void setNumTombstones(unsigned Num) { NumTombstones = Num; }

  BucketT *getBuckets() const { return Buckets; }

  unsigned getNumBuckets() const { return NumBuckets; }

  bool allocateBuckets(unsigned Num) {
    NumBuckets = Num;
    if (NumBuckets == 0) {
      Buckets = nullptr;
      return false;
    }

    Buckets = static_cast<BucketT *>(
        allocate_buffer(sizeof(BucketT) * NumBuckets, alignof(BucketT)));
    return true;
  }
};

}  // namespace __sanitizer

#endif  // SANITIZER_DENSE_MAP_H
