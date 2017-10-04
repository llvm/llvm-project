//===-- xray_buffer_queue.cc -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instruementation system.
//
// Defines the interface for a buffer queue implementation.
//
//===----------------------------------------------------------------------===//
#include "xray_buffer_queue.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <algorithm>
#include <cstdlib>
#include <tuple>

using namespace __xray;
using namespace __sanitizer;

BufferQueue::BufferQueue(std::size_t B, std::size_t N, bool &Success)
    : BufferSize(B), Buffers(new std::tuple<Buffer, bool>[N]()),
      BufferCount(N), Finalizing{0}, OwnedBuffers(new void *[N]()),
      Next(Buffers.get()), First(Buffers.get()), LiveBuffers(0) {
  for (size_t i = 0; i < N; ++i) {
    auto &T = Buffers[i];
    void *Tmp = malloc(BufferSize);
    if (Tmp == nullptr) {
      Success = false;
      return;
    }
    auto &Buf = std::get<0>(T);
    std::get<1>(T) = false;
    Buf.Buffer = Tmp;
    Buf.Size = B;
    OwnedBuffers[i] = Tmp;
  }
  Success = true;
}

BufferQueue::ErrorCode BufferQueue::getBuffer(Buffer &Buf) {
  if (__sanitizer::atomic_load(&Finalizing, __sanitizer::memory_order_acquire))
    return ErrorCode::QueueFinalizing;
  __sanitizer::SpinMutexLock Guard(&Mutex);
  if (LiveBuffers == BufferCount)
    return ErrorCode::NotEnoughMemory;

  auto &T = *Next;
  auto &B = std::get<0>(T);
  Buf = B;
  ++LiveBuffers;

  if (++Next == (Buffers.get() + BufferCount))
    Next = Buffers.get();

  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::releaseBuffer(Buffer &Buf) {
  // Blitz through the buffers array to find the buffer.
  if (std::none_of(OwnedBuffers.get(), OwnedBuffers.get() + BufferCount,
                   [&Buf](void *P) { return P == Buf.Buffer; }))
    return ErrorCode::UnrecognizedBuffer;
  __sanitizer::SpinMutexLock Guard(&Mutex);

  // This points to a semantic bug, we really ought to not be releasing more
  // buffers than we actually get.
  if (LiveBuffers == 0)
    return ErrorCode::NotEnoughMemory;

  // Now that the buffer has been released, we mark it as "used".
  *First = std::make_tuple(Buf, true);
  Buf.Buffer = nullptr;
  Buf.Size = 0;
  --LiveBuffers;
  if (++First == (Buffers.get() + BufferCount))
    First = Buffers.get();

  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::finalize() {
  if (__sanitizer::atomic_exchange(&Finalizing, 1,
                                   __sanitizer::memory_order_acq_rel))
    return ErrorCode::QueueFinalizing;
  return ErrorCode::Ok;
}

BufferQueue::~BufferQueue() {
  for (auto I = Buffers.get(), E = Buffers.get() + BufferCount; I != E; ++I) {
    auto &T = *I;
    auto &Buf = std::get<0>(T);
    free(Buf.Buffer);
  }
}
