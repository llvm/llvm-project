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
#include "sanitizer_common/sanitizer_mutex.h"
#include <deque>
#include <unordered_set>
#include <utility>

namespace __xray {

/// BufferQueue implements a circular queue of fixed sized buffers (much like a
/// freelist) but is concerned mostly with making it really quick to initialise,
/// finalise, and get/return buffers to the queue. This is one key component of
/// the "flight data recorder" (FDR) mode to support ongoing XRay function call
/// trace collection.
class BufferQueue {
public:
  struct Buffer {
    void *Buffer = nullptr;
    size_t Size = 0;
  };

private:
  size_t BufferSize;

  // We use a bool to indicate whether the Buffer has been used in this
  // freelist implementation.
  std::deque<std::tuple<Buffer, bool>> Buffers;
  __sanitizer::BlockingMutex Mutex;
  std::unordered_set<void *> OwnedBuffers;
  __sanitizer::atomic_uint8_t Finalizing;

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
    __sanitizer::BlockingMutexLock G(&Mutex);
    for (const auto &T : Buffers) {
      if (std::get<1>(T))
        Fn(std::get<0>(T));
    }
  }

  // Cleans up allocated buffers.
  ~BufferQueue();
};

} // namespace __xray

#endif // XRAY_BUFFER_QUEUE_H
