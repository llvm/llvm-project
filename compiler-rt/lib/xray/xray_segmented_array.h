//===-- xray_segmented_array.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Defines the implementation of a segmented array, with fixed-size chunks
// backing the segments.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_SEGMENTED_ARRAY_H
#define XRAY_SEGMENTED_ARRAY_H

#include "sanitizer_common/sanitizer_allocator.h"
#include "xray_allocator.h"
#include "xray_utils.h"
#include <cassert>
#include <type_traits>
#include <utility>

namespace __xray {

/// The Array type provides an interface similar to std::vector<...> but does
/// not shrink in size. Once constructed, elements can be appended but cannot be
/// removed. The implementation is heavily dependent on the contract provided by
/// the Allocator type, in that all memory will be released when the Allocator
/// is destroyed. When an Array is destroyed, it will destroy elements in the
/// backing store but will not free the memory. The parameter N defines how many
/// elements of T there should be in a single block.
///
/// We compute the least common multiple of the size of T and the cache line
/// size, to allow us to maximise the number of T objects we can place in
/// cache-line multiple sized blocks. To get back the number of T's, we divide
/// this least common multiple by the size of T.
template <class T,
          size_t N = lcm(next_pow2(sizeof(T)), kCacheLineSize) / sizeof(T)>
struct Array {
  static constexpr size_t ChunkSize = N;
  static constexpr size_t ElementStorageSize = next_pow2(sizeof(T));
  static constexpr size_t AllocatorChunkSize = ElementStorageSize * ChunkSize;
  using AllocatorType = Allocator<AllocatorChunkSize>;
  static_assert(std::is_trivially_destructible<T>::value,
                "T must be trivially destructible.");

private:
  // TODO: Consider co-locating the chunk information with the data in the
  // Block, as in an intrusive list -- i.e. putting the next and previous
  // pointer values inside the Block storage.
  struct Chunk {
    typename AllocatorType::Block Block;
    static constexpr size_t Size = N;
    Chunk *Prev = this;
    Chunk *Next = this;
  };

  static Chunk SentinelChunk;

  AllocatorType *Alloc;
  Chunk *Head = &SentinelChunk;
  Chunk *Tail = &SentinelChunk;
  size_t Size = 0;

  // Here we keep track of chunks in the freelist, to allow us to re-use chunks
  // when elements are trimmed off the end.
  Chunk *Freelist = &SentinelChunk;

  Chunk *NewChunk() {
    // We need to handle the case in which enough elements have been trimmed to
    // allow us to re-use chunks we've allocated before. For this we look into
    // the Freelist, to see whether we need to actually allocate new blocks or
    // just re-use blocks we've already seen before.
    if (Freelist != &SentinelChunk) {
      auto *FreeChunk = Freelist;
      Freelist = FreeChunk->Next;
      FreeChunk->Next = &SentinelChunk;
      Freelist->Prev = &SentinelChunk;
      return FreeChunk;
    }

    auto Block = Alloc->Allocate(ElementStorageSize);
    if (Block.Data == nullptr)
      return nullptr;

    // TODO: Maybe use a separate managed allocator for Chunk instances?
    auto C = reinterpret_cast<Chunk *>(
        InternalAlloc(sizeof(Chunk), nullptr, kCacheLineSize));
    if (C == nullptr)
      return nullptr;
    C->Block = Block;
    C->Prev = &SentinelChunk;
    C->Next = &SentinelChunk;
    return C;
  }

  static AllocatorType &GetGlobalAllocator() {
    static AllocatorType *const GlobalAllocator = [] {
      AllocatorType *A = reinterpret_cast<AllocatorType *>(
          InternalAlloc(sizeof(AllocatorType)));
      new (A) AllocatorType(2 << 10, 0);
      return A;
    }();

    return *GlobalAllocator;
  }

  Chunk *InitHeadAndTail() {
    DCHECK_EQ(Head, &SentinelChunk);
    DCHECK_EQ(Tail, &SentinelChunk);
    auto Chunk = NewChunk();
    if (Chunk == nullptr)
      return nullptr;
    DCHECK_EQ(Chunk->Next, &SentinelChunk);
    DCHECK_EQ(Chunk->Prev, &SentinelChunk);
    return Head = Tail = Chunk;
  }

  Chunk *AppendNewChunk() {
    auto Chunk = NewChunk();
    if (Chunk == nullptr)
      return nullptr;
    DCHECK_NE(Tail, &SentinelChunk);
    DCHECK_EQ(Tail->Next, &SentinelChunk);
    DCHECK_EQ(Chunk->Prev, &SentinelChunk);
    DCHECK_EQ(Chunk->Next, &SentinelChunk);
    Tail->Next = Chunk;
    Chunk->Prev = Tail;
    Tail = Chunk;
    return Tail;
  }

  // This Iterator models a BidirectionalIterator.
  template <class U> class Iterator {
    Chunk *C = &SentinelChunk;
    size_t Offset = 0;
    size_t Size = 0;

  public:
    Iterator(Chunk *IC, size_t Off, size_t S) : C(IC), Offset(Off), Size(S) {}
    Iterator(const Iterator &) noexcept = default;
    Iterator() noexcept = default;
    Iterator(Iterator &&) noexcept = default;
    Iterator &operator=(const Iterator &) = default;
    Iterator &operator=(Iterator &&) = default;
    ~Iterator() = default;

    Iterator &operator++() {
      if (++Offset % N || Offset == Size)
        return *this;

      // At this point, we know that Offset % N == 0, so we must advance the
      // chunk pointer.
      DCHECK_EQ(Offset % N, 0);
      DCHECK_NE(Offset, Size);
      DCHECK_NE(C, &SentinelChunk);
      DCHECK_NE(C->Next, &SentinelChunk);
      C = C->Next;
      DCHECK_NE(C, &SentinelChunk);
      return *this;
    }

    Iterator &operator--() {
      DCHECK_NE(C, &SentinelChunk);
      DCHECK_GT(Offset, 0);

      auto PreviousOffset = Offset--;
      if (PreviousOffset != Size && PreviousOffset % N == 0) {
        DCHECK_NE(C->Prev, &SentinelChunk);
        C = C->Prev;
      }

      return *this;
    }

    Iterator operator++(int) {
      Iterator Copy(*this);
      ++(*this);
      return Copy;
    }

    Iterator operator--(int) {
      Iterator Copy(*this);
      --(*this);
      return Copy;
    }

    template <class V, class W>
    friend bool operator==(const Iterator<V> &L, const Iterator<W> &R) {
      return L.C == R.C && L.Offset == R.Offset;
    }

    template <class V, class W>
    friend bool operator!=(const Iterator<V> &L, const Iterator<W> &R) {
      return !(L == R);
    }

    U &operator*() const {
      DCHECK_NE(C, &SentinelChunk);
      auto RelOff = Offset % N;
      return *reinterpret_cast<U *>(reinterpret_cast<char *>(C->Block.Data) +
                                    (RelOff * ElementStorageSize));
    }

    U *operator->() const {
      DCHECK_NE(C, &SentinelChunk);
      auto RelOff = Offset % N;
      return reinterpret_cast<U *>(reinterpret_cast<char *>(C->Block.Data) +
                                   (RelOff * ElementStorageSize));
    }
  };

public:
  explicit Array(AllocatorType &A) : Alloc(&A) {}
  Array() : Array(GetGlobalAllocator()) {}

  Array(const Array &) = delete;
  Array(Array &&O) NOEXCEPT : Alloc(O.Alloc),
                              Head(O.Head),
                              Tail(O.Tail),
                              Size(O.Size) {
    O.Head = &SentinelChunk;
    O.Tail = &SentinelChunk;
    O.Size = 0;
  }

  bool empty() const { return Size == 0; }

  AllocatorType &allocator() const {
    DCHECK_NE(Alloc, nullptr);
    return *Alloc;
  }

  size_t size() const { return Size; }

  T *Append(const T &E) {
    if (UNLIKELY(Head == &SentinelChunk))
      if (InitHeadAndTail() == nullptr)
        return nullptr;

    auto Offset = Size % N;
    if (UNLIKELY(Size != 0 && Offset == 0))
      if (AppendNewChunk() == nullptr)
        return nullptr;

    auto Position =
        reinterpret_cast<T *>(reinterpret_cast<char *>(Tail->Block.Data) +
                              (Offset * ElementStorageSize));
    *Position = E;
    ++Size;
    return Position;
  }

  template <class... Args> T *AppendEmplace(Args &&... args) {
    if (UNLIKELY(Head == &SentinelChunk))
      if (InitHeadAndTail() == nullptr)
        return nullptr;

    auto Offset = Size % N;
    auto *LatestChunk = Tail;
    if (UNLIKELY(Size != 0 && Offset == 0)) {
      LatestChunk = AppendNewChunk();
      if (LatestChunk == nullptr)
        return nullptr;
    }

    DCHECK_NE(Tail, &SentinelChunk);
    auto Position = reinterpret_cast<char *>(LatestChunk->Block.Data) +
                    (Offset * ElementStorageSize);
    DCHECK_EQ(reinterpret_cast<uintptr_t>(Position) % ElementStorageSize, 0);
    // In-place construct at Position.
    new (Position) T{std::forward<Args>(args)...};
    ++Size;
    return reinterpret_cast<T *>(Position);
  }

  T &operator[](size_t Offset) const {
    DCHECK_LE(Offset, Size);
    // We need to traverse the array enough times to find the element at Offset.
    auto C = Head;
    while (Offset >= N) {
      C = C->Next;
      Offset -= N;
      DCHECK_NE(C, &SentinelChunk);
    }
    return *reinterpret_cast<T *>(reinterpret_cast<char *>(C->Block.Data) +
                                  (Offset * ElementStorageSize));
  }

  T &front() const {
    DCHECK_NE(Head, &SentinelChunk);
    DCHECK_NE(Size, 0u);
    return *begin();
  }

  T &back() const {
    DCHECK_NE(Tail, &SentinelChunk);
    DCHECK_NE(Size, 0u);
    auto It = end();
    --It;
    return *It;
  }

  template <class Predicate> T *find_element(Predicate P) const {
    if (empty())
      return nullptr;

    auto E = end();
    for (auto I = begin(); I != E; ++I)
      if (P(*I))
        return &(*I);

    return nullptr;
  }

  /// Remove N Elements from the end. This leaves the blocks behind, and not
  /// require allocation of new blocks for new elements added after trimming.
  void trim(size_t Elements) {
    DCHECK_LE(Elements, Size);
    DCHECK_GT(Size, 0);
    auto OldSize = Size;
    Size -= Elements;

    DCHECK_NE(Head, &SentinelChunk);
    DCHECK_NE(Tail, &SentinelChunk);

    for (auto ChunksToTrim =
             (nearest_boundary(OldSize, N) - nearest_boundary(Size, N)) / N;
         ChunksToTrim > 0; --ChunksToTrim) {
      DCHECK_NE(Head, &SentinelChunk);
      DCHECK_NE(Tail, &SentinelChunk);
      // Put the tail into the Freelist.
      auto *FreeChunk = Tail;
      Tail = Tail->Prev;
      if (Tail == &SentinelChunk)
        Head = Tail;
      else
        Tail->Next = &SentinelChunk;

      DCHECK_EQ(Tail->Next, &SentinelChunk);
      FreeChunk->Next = Freelist;
      FreeChunk->Prev = &SentinelChunk;
      if (Freelist != &SentinelChunk)
        Freelist->Prev = FreeChunk;
      Freelist = FreeChunk;
    }
  }

  // Provide iterators.
  Iterator<T> begin() const { return Iterator<T>(Head, 0, Size); }
  Iterator<T> end() const { return Iterator<T>(Tail, Size, Size); }
  Iterator<const T> cbegin() const { return Iterator<const T>(Head, 0, Size); }
  Iterator<const T> cend() const { return Iterator<const T>(Tail, Size, Size); }
};

// We need to have this storage definition out-of-line so that the compiler can
// ensure that storage for the SentinelChunk is defined and has a single
// address.
template <class T, size_t N>
typename Array<T, N>::Chunk Array<T, N>::SentinelChunk;

} // namespace __xray

#endif // XRAY_SEGMENTED_ARRAY_H
