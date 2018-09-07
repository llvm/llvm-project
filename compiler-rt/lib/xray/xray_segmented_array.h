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
// Defines the implementation of a segmented array, with fixed-size segments
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
/// backing store but will not free the memory.
template <class T> class Array {
  struct SegmentBase {
    SegmentBase *Prev;
    SegmentBase *Next;
  };

  // We want each segment of the array to be cache-line aligned, and elements of
  // the array be offset from the beginning of the segment.
  struct Segment : SegmentBase {
    char Data[1];
  };

public:
  // Each segment of the array will be laid out with the following assumptions:
  //
  //   - Each segment will be on a cache-line address boundary (kCacheLineSize
  //     aligned).
  //
  //   - The elements will be accessed through an aligned pointer, dependent on
  //     the alignment of T.
  //
  //   - Each element is at least two-pointers worth from the beginning of the
  //     Segment, aligned properly, and the rest of the elements are accessed
  //     through appropriate alignment.
  //
  // We then compute the size of the segment to follow this logic:
  //
  //   - Compute the number of elements that can fit within
  //     kCacheLineSize-multiple segments, minus the size of two pointers.
  //
  //   - Request cacheline-multiple sized elements from the allocator.
  static constexpr size_t AlignedElementStorageSize =
      sizeof(typename std::aligned_storage<sizeof(T), alignof(T)>::type);

  static constexpr size_t SegmentSize =
      nearest_boundary(sizeof(Segment) + next_pow2(sizeof(T)), kCacheLineSize);

  using AllocatorType = Allocator<SegmentSize>;

  static constexpr size_t ElementsPerSegment =
      (SegmentSize - sizeof(Segment)) / next_pow2(sizeof(T));

  static_assert(ElementsPerSegment > 0,
                "Must have at least 1 element per segment.");

  static SegmentBase SentinelSegment;

private:
  AllocatorType *Alloc;
  SegmentBase *Head = &SentinelSegment;
  SegmentBase *Tail = &SentinelSegment;
  size_t Size = 0;

  // Here we keep track of segments in the freelist, to allow us to re-use
  // segments when elements are trimmed off the end.
  SegmentBase *Freelist = &SentinelSegment;

  Segment *NewSegment() XRAY_NEVER_INSTRUMENT {
    // We need to handle the case in which enough elements have been trimmed to
    // allow us to re-use segments we've allocated before. For this we look into
    // the Freelist, to see whether we need to actually allocate new blocks or
    // just re-use blocks we've already seen before.
    if (Freelist != &SentinelSegment) {
      auto *FreeSegment = Freelist;
      Freelist = FreeSegment->Next;
      FreeSegment->Next = &SentinelSegment;
      Freelist->Prev = &SentinelSegment;
      return static_cast<Segment *>(FreeSegment);
    }

    auto SegmentBlock = Alloc->Allocate();
    if (SegmentBlock.Data == nullptr)
      return nullptr;

    // Placement-new the Segment element at the beginning of the SegmentBlock.
    auto S = reinterpret_cast<Segment *>(SegmentBlock.Data);
    new (S) SegmentBase{&SentinelSegment, &SentinelSegment};
    return S;
  }

  Segment *InitHeadAndTail() XRAY_NEVER_INSTRUMENT {
    DCHECK_EQ(Head, &SentinelSegment);
    DCHECK_EQ(Tail, &SentinelSegment);
    auto Segment = NewSegment();
    if (Segment == nullptr)
      return nullptr;
    DCHECK_EQ(Segment->Next, &SentinelSegment);
    DCHECK_EQ(Segment->Prev, &SentinelSegment);
    Head = Tail = static_cast<SegmentBase *>(Segment);
    return Segment;
  }

  Segment *AppendNewSegment() XRAY_NEVER_INSTRUMENT {
    auto S = NewSegment();
    if (S == nullptr)
      return nullptr;
    DCHECK_NE(Tail, &SentinelSegment);
    DCHECK_EQ(Tail->Next, &SentinelSegment);
    DCHECK_EQ(S->Prev, &SentinelSegment);
    DCHECK_EQ(S->Next, &SentinelSegment);
    Tail->Next = S;
    S->Prev = Tail;
    Tail = S;
    return static_cast<Segment *>(Tail);
  }

  // This Iterator models a BidirectionalIterator.
  template <class U> class Iterator {
    SegmentBase *S = &SentinelSegment;
    size_t Offset = 0;
    size_t Size = 0;

  public:
    Iterator(SegmentBase *IS, size_t Off, size_t S) XRAY_NEVER_INSTRUMENT
        : S(IS),
          Offset(Off),
          Size(S) {}
    Iterator(const Iterator &) NOEXCEPT XRAY_NEVER_INSTRUMENT = default;
    Iterator() NOEXCEPT XRAY_NEVER_INSTRUMENT = default;
    Iterator(Iterator &&) NOEXCEPT XRAY_NEVER_INSTRUMENT = default;
    Iterator &operator=(const Iterator &) XRAY_NEVER_INSTRUMENT = default;
    Iterator &operator=(Iterator &&) XRAY_NEVER_INSTRUMENT = default;
    ~Iterator() XRAY_NEVER_INSTRUMENT = default;

    Iterator &operator++() XRAY_NEVER_INSTRUMENT {
      if (++Offset % ElementsPerSegment || Offset == Size)
        return *this;

      // At this point, we know that Offset % N == 0, so we must advance the
      // segment pointer.
      DCHECK_EQ(Offset % ElementsPerSegment, 0);
      DCHECK_NE(Offset, Size);
      DCHECK_NE(S, &SentinelSegment);
      DCHECK_NE(S->Next, &SentinelSegment);
      S = S->Next;
      DCHECK_NE(S, &SentinelSegment);
      return *this;
    }

    Iterator &operator--() XRAY_NEVER_INSTRUMENT {
      DCHECK_NE(S, &SentinelSegment);
      DCHECK_GT(Offset, 0);

      auto PreviousOffset = Offset--;
      if (PreviousOffset != Size && PreviousOffset % ElementsPerSegment == 0) {
        DCHECK_NE(S->Prev, &SentinelSegment);
        S = S->Prev;
      }

      return *this;
    }

    Iterator operator++(int) XRAY_NEVER_INSTRUMENT {
      Iterator Copy(*this);
      ++(*this);
      return Copy;
    }

    Iterator operator--(int) XRAY_NEVER_INSTRUMENT {
      Iterator Copy(*this);
      --(*this);
      return Copy;
    }

    template <class V, class W>
    friend bool operator==(const Iterator<V> &L,
                           const Iterator<W> &R) XRAY_NEVER_INSTRUMENT {
      return L.S == R.S && L.Offset == R.Offset;
    }

    template <class V, class W>
    friend bool operator!=(const Iterator<V> &L,
                           const Iterator<W> &R) XRAY_NEVER_INSTRUMENT {
      return !(L == R);
    }

    U &operator*() const XRAY_NEVER_INSTRUMENT {
      DCHECK_NE(S, &SentinelSegment);
      auto RelOff = Offset % ElementsPerSegment;

      // We need to compute the character-aligned pointer, offset from the
      // segment's Data location to get the element in the position of Offset.
      auto Base = static_cast<Segment *>(S)->Data;
      auto AlignedOffset = Base + (RelOff * AlignedElementStorageSize);
      return *reinterpret_cast<U *>(AlignedOffset);
    }

    U *operator->() const XRAY_NEVER_INSTRUMENT { return &(**this); }
  };

public:
  explicit Array(AllocatorType &A) XRAY_NEVER_INSTRUMENT : Alloc(&A) {}

  Array(const Array &) = delete;
  Array(Array &&O) NOEXCEPT : Alloc(O.Alloc),
                              Head(O.Head),
                              Tail(O.Tail),
                              Size(O.Size) {
    O.Head = &SentinelSegment;
    O.Tail = &SentinelSegment;
    O.Size = 0;
  }

  bool empty() const XRAY_NEVER_INSTRUMENT { return Size == 0; }

  AllocatorType &allocator() const XRAY_NEVER_INSTRUMENT {
    DCHECK_NE(Alloc, nullptr);
    return *Alloc;
  }

  size_t size() const XRAY_NEVER_INSTRUMENT { return Size; }

  T *Append(const T &E) XRAY_NEVER_INSTRUMENT {
    if (UNLIKELY(Head == &SentinelSegment))
      if (InitHeadAndTail() == nullptr)
        return nullptr;

    auto Offset = Size % ElementsPerSegment;
    if (UNLIKELY(Size != 0 && Offset == 0))
      if (AppendNewSegment() == nullptr)
        return nullptr;

    auto Base = static_cast<Segment *>(Tail)->Data;
    auto AlignedOffset = Base + (Offset * AlignedElementStorageSize);
    auto Position = reinterpret_cast<T *>(AlignedOffset);
    *Position = E;
    ++Size;
    return Position;
  }

  template <class... Args>
  T *AppendEmplace(Args &&... args) XRAY_NEVER_INSTRUMENT {
    if (UNLIKELY(Head == &SentinelSegment))
      if (InitHeadAndTail() == nullptr)
        return nullptr;

    auto Offset = Size % ElementsPerSegment;
    auto *LatestSegment = Tail;
    if (UNLIKELY(Size != 0 && Offset == 0)) {
      LatestSegment = AppendNewSegment();
      if (LatestSegment == nullptr)
        return nullptr;
    }

    DCHECK_NE(Tail, &SentinelSegment);
    auto Base = static_cast<Segment *>(LatestSegment)->Data;
    auto AlignedOffset = Base + (Offset * AlignedElementStorageSize);
    auto Position = reinterpret_cast<T *>(AlignedOffset);

    // In-place construct at Position.
    new (Position) T{std::forward<Args>(args)...};
    ++Size;
    return reinterpret_cast<T *>(Position);
  }

  T &operator[](size_t Offset) const XRAY_NEVER_INSTRUMENT {
    DCHECK_LE(Offset, Size);
    // We need to traverse the array enough times to find the element at Offset.
    auto S = Head;
    while (Offset >= ElementsPerSegment) {
      S = S->Next;
      Offset -= ElementsPerSegment;
      DCHECK_NE(S, &SentinelSegment);
    }
    auto Base = static_cast<Segment *>(S)->Data;
    auto AlignedOffset = Base + (Offset * AlignedElementStorageSize);
    auto Position = reinterpret_cast<T *>(AlignedOffset);
    return *reinterpret_cast<T *>(Position);
  }

  T &front() const XRAY_NEVER_INSTRUMENT {
    DCHECK_NE(Head, &SentinelSegment);
    DCHECK_NE(Size, 0u);
    return *begin();
  }

  T &back() const XRAY_NEVER_INSTRUMENT {
    DCHECK_NE(Tail, &SentinelSegment);
    DCHECK_NE(Size, 0u);
    auto It = end();
    --It;
    return *It;
  }

  template <class Predicate>
  T *find_element(Predicate P) const XRAY_NEVER_INSTRUMENT {
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
  void trim(size_t Elements) XRAY_NEVER_INSTRUMENT {
    if (Elements == 0)
      return;

    DCHECK_LE(Elements, Size);
    DCHECK_GT(Size, 0);
    auto OldSize = Size;
    Size -= Elements;

    DCHECK_NE(Head, &SentinelSegment);
    DCHECK_NE(Tail, &SentinelSegment);

    for (auto SegmentsToTrim = (nearest_boundary(OldSize, ElementsPerSegment) -
                                nearest_boundary(Size, ElementsPerSegment)) /
                               ElementsPerSegment;
         SegmentsToTrim > 0; --SegmentsToTrim) {
      DCHECK_NE(Head, &SentinelSegment);
      DCHECK_NE(Tail, &SentinelSegment);
      // Put the tail into the Freelist.
      auto *FreeSegment = Tail;
      Tail = Tail->Prev;
      if (Tail == &SentinelSegment)
        Head = Tail;
      else
        Tail->Next = &SentinelSegment;

      DCHECK_EQ(Tail->Next, &SentinelSegment);
      FreeSegment->Next = Freelist;
      FreeSegment->Prev = &SentinelSegment;
      if (Freelist != &SentinelSegment)
        Freelist->Prev = FreeSegment;
      Freelist = FreeSegment;
    }
  }

  // Provide iterators.
  Iterator<T> begin() const XRAY_NEVER_INSTRUMENT {
    return Iterator<T>(Head, 0, Size);
  }
  Iterator<T> end() const XRAY_NEVER_INSTRUMENT {
    return Iterator<T>(Tail, Size, Size);
  }
  Iterator<const T> cbegin() const XRAY_NEVER_INSTRUMENT {
    return Iterator<const T>(Head, 0, Size);
  }
  Iterator<const T> cend() const XRAY_NEVER_INSTRUMENT {
    return Iterator<const T>(Tail, Size, Size);
  }
};

// We need to have this storage definition out-of-line so that the compiler can
// ensure that storage for the SentinelSegment is defined and has a single
// address.
template <class T>
typename Array<T>::SegmentBase Array<T>::SentinelSegment{
    &Array<T>::SentinelSegment, &Array<T>::SentinelSegment};

} // namespace __xray

#endif // XRAY_SEGMENTED_ARRAY_H
