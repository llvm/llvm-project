//===-- sanitizer_ring_buffer.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple ring buffer.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_RING_BUFFER_H
#define SANITIZER_RING_BUFFER_H

#include "sanitizer_common.h"

namespace __sanitizer {
// RingBuffer<T>: fixed-size ring buffer optimized for speed of push().
// T should be a POD type and sizeof(T) should be divisible by sizeof(void*).
// At creation, all elements are zero.
template<class T>
class RingBuffer {
 public:
  COMPILER_CHECK(sizeof(T) % sizeof(void *) == 0);
  static RingBuffer *New(uptr Size) {
    void *Ptr = MmapOrDie(SizeInBytes(Size), "RingBuffer");
    RingBuffer *RB = reinterpret_cast<RingBuffer*>(Ptr);
    uptr End = reinterpret_cast<uptr>(Ptr) + SizeInBytes(Size);
    RB->last_ = RB->next_ = reinterpret_cast<T*>(End - sizeof(T));
    return RB;
  }
  void Delete() {
    UnmapOrDie(this, SizeInBytes(size()));
  }
  uptr size() const {
    return last_ + 1 -
           reinterpret_cast<T *>(reinterpret_cast<uptr>(this) +
                                 2 * sizeof(T *));
  }

  uptr SizeInBytes() { return SizeInBytes(size()); }

  void push(T t) {
    *next_ = t;
    next_--;
    // The condition below works only if sizeof(T) is divisible by sizeof(T*).
    if (next_ <= reinterpret_cast<T*>(&next_))
      next_ = last_;
  }

  T operator[](uptr Idx) const {
    CHECK_LT(Idx, size());
    sptr IdxNext = Idx + 1;
    if (IdxNext > last_ - next_)
      IdxNext -= size();
    return next_[IdxNext];
  }

 private:
  RingBuffer() {}
  ~RingBuffer() {}
  RingBuffer(const RingBuffer&) = delete;

  static uptr SizeInBytes(uptr Size) {
    return Size * sizeof(T) + 2 * sizeof(T*);
  }

  // Data layout:
  // LNDDDDDDDD
  // D: data elements.
  // L: last_, always points to the last data element.
  // N: next_, initially equals to last_, is decremented on every push,
  //    wraps around if it's less or equal than its own address.

  T *last_;
  T *next_;
  T data_[1];  // flexible array.
};

}  // namespace __sanitizer

#endif  // SANITIZER_RING_BUFFER_H
