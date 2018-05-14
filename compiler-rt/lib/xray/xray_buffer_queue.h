//===-- xray_buffer_queue.h ------------------------------------*- C++ -*-===//
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
// Defines the interface for a buffer queue implementation.
//
//===----------------------------------------------------------------------===//
#ifndef XRAY_BUFFER_QUEUE_H
#define XRAY_BUFFER_QUEUE_H

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include <cstddef>

namespace __xray {

/// BufferQueue implements a circular queue of fixed sized buffers (much like a
/// freelist) but is concerned mostly with making it really quick to initialise,
/// finalise, and get/return buffers to the queue. This is one key component of
/// the "flight data recorder" (FDR) mode to support ongoing XRay function call
/// trace collection.
class BufferQueue {
public:
  struct alignas(64) BufferExtents {
    __sanitizer::atomic_uint64_t Size;
  };

  struct Buffer {
    void *Data = nullptr;
    size_t Size = 0;
    BufferExtents *Extents;
  };

private:
  struct BufferRep {
    // The managed buffer.
    Buffer Buff;

    // This is true if the buffer has been returned to the available queue, and
    // is considered "used" by another thread.
    bool Used = false;
  };

  // This models a ForwardIterator. |T| Must be either a `Buffer` or `const
  // Buffer`. Note that we only advance to the "used" buffers, when
  // incrementing, so that at dereference we're always at a valid point.
  template <class T> class Iterator {
  public:
    BufferRep *Buffers = nullptr;
    size_t Offset = 0;
    size_t Max = 0;

    Iterator &operator++() {
      DCHECK_NE(Offset, Max);
      do {
        ++Offset;
      } while (!Buffers[Offset].Used && Offset != Max);
      return *this;
    }

    Iterator operator++(int) {
      Iterator C = *this;
      ++(*this);
      return C;
    }

    T &operator*() const { return Buffers[Offset].Buff; }

    T *operator->() const { return &(Buffers[Offset].Buff); }

    Iterator(BufferRep *Root, size_t O, size_t M)
        : Buffers(Root), Offset(O), Max(M) {
      // We want to advance to the first Offset where the 'Used' property is
      // true, or to the end of the list/queue.
      while (!Buffers[Offset].Used && Offset != Max) {
        ++Offset;
      }
    }

    Iterator() = default;
    Iterator(const Iterator &) = default;
    Iterator(Iterator &&) = default;
    Iterator &operator=(const Iterator &) = default;
    Iterator &operator=(Iterator &&) = default;
    ~Iterator() = default;

    template <class V>
    friend bool operator==(const Iterator &L, const Iterator<V> &R) {
      DCHECK_EQ(L.Max, R.Max);
      return L.Buffers == R.Buffers && L.Offset == R.Offset;
    }

    template <class V>
    friend bool operator!=(const Iterator &L, const Iterator<V> &R) {
      return !(L == R);
    }
  };

  // Size of each individual Buffer.
  size_t BufferSize;

  BufferRep *Buffers;

  // Amount of pre-allocated buffers.
  size_t BufferCount;

  __sanitizer::SpinMutex Mutex;
  __sanitizer::atomic_uint8_t Finalizing;

  // Pointers to buffers managed/owned by the BufferQueue.
  void **OwnedBuffers;

  // Pointer to the next buffer to be handed out.
  BufferRep *Next;

  // Pointer to the entry in the array where the next released buffer will be
  // placed.
  BufferRep *First;

  // Count of buffers that have been handed out through 'getBuffer'.
  size_t LiveBuffers;

public:
  enum class ErrorCode : unsigned {
    Ok,
    NotEnoughMemory,
    QueueFinalizing,
    UnrecognizedBuffer,
    AlreadyFinalized,
  };

  static const char *getErrorString(ErrorCode E) {
    switch (E) {
    case ErrorCode::Ok:
      return "(none)";
    case ErrorCode::NotEnoughMemory:
      return "no available buffers in the queue";
    case ErrorCode::QueueFinalizing:
      return "queue already finalizing";
    case ErrorCode::UnrecognizedBuffer:
      return "buffer being returned not owned by buffer queue";
    case ErrorCode::AlreadyFinalized:
      return "queue already finalized";
    }
    return "unknown error";
  }

  /// Initialise a queue of size |N| with buffers of size |B|. We report success
  /// through |Success|.
  BufferQueue(size_t B, size_t N, bool &Success);

  /// Updates |Buf| to contain the pointer to an appropriate buffer. Returns an
  /// error in case there are no available buffers to return when we will run
  /// over the upper bound for the total buffers.
  ///
  /// Requirements:
  ///   - BufferQueue is not finalising.
  ///
  /// Returns:
  ///   - ErrorCode::NotEnoughMemory on exceeding MaxSize.
  ///   - ErrorCode::Ok when we find a Buffer.
  ///   - ErrorCode::QueueFinalizing or ErrorCode::AlreadyFinalized on
  ///     a finalizing/finalized BufferQueue.
  ErrorCode getBuffer(Buffer &Buf);

  /// Updates |Buf| to point to nullptr, with size 0.
  ///
  /// Returns:
  ///   - ErrorCode::Ok when we successfully release the buffer.
  ///   - ErrorCode::UnrecognizedBuffer for when this BufferQueue does not own
  ///     the buffer being released.
  ErrorCode releaseBuffer(Buffer &Buf);

  bool finalizing() const {
    return __sanitizer::atomic_load(&Finalizing,
                                    __sanitizer::memory_order_acquire);
  }

  /// Returns the configured size of the buffers in the buffer queue.
  size_t ConfiguredBufferSize() const { return BufferSize; }

  /// Sets the state of the BufferQueue to finalizing, which ensures that:
  ///
  ///   - All subsequent attempts to retrieve a Buffer will fail.
  ///   - All releaseBuffer operations will not fail.
  ///
  /// After a call to finalize succeeds, all subsequent calls to finalize will
  /// fail with ErrorCode::QueueFinalizing.
  ErrorCode finalize();

  /// Applies the provided function F to each Buffer in the queue, only if the
  /// Buffer is marked 'used' (i.e. has been the result of getBuffer(...) and a
  /// releaseBuffer(...) operation).
  template <class F> void apply(F Fn) {
    __sanitizer::SpinMutexLock G(&Mutex);
    for (auto I = begin(), E = end(); I != E; ++I)
      Fn(*I);
  }

  using const_iterator = Iterator<const Buffer>;
  using iterator = Iterator<Buffer>;

  /// Provides iterator access to the raw Buffer instances.
  iterator begin() const { return iterator(Buffers, 0, BufferCount); }
  const_iterator cbegin() const {
    return const_iterator(Buffers, 0, BufferCount);
  }
  iterator end() const { return iterator(Buffers, BufferCount, BufferCount); }
  const_iterator cend() const {
    return const_iterator(Buffers, BufferCount, BufferCount);
  }

  // Cleans up allocated buffers.
  ~BufferQueue();
};

} // namespace __xray

#endif // XRAY_BUFFER_QUEUE_H
