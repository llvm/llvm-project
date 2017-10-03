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

#include <cstdlib>
#include <tuple>

using namespace __xray;
using namespace __sanitizer;

BufferQueue::BufferQueue(std::size_t B, std::size_t N, bool &Success)
    : BufferSize(B), Buffers(N), Mutex(), OwnedBuffers(), Finalizing{0} {
  for (auto &T : Buffers) {
    void *Tmp = malloc(BufferSize);
    if (Tmp == nullptr) {
      Success = false;
      return;
    }

    auto &Buf = std::get<0>(T);
    Buf.Buffer = Tmp;
    Buf.Size = B;
    OwnedBuffers.emplace(Tmp);
  }
  Success = true;
}

BufferQueue::ErrorCode BufferQueue::getBuffer(Buffer &Buf) {
  if (__sanitizer::atomic_load(&Finalizing, __sanitizer::memory_order_acquire))
    return ErrorCode::QueueFinalizing;
  __sanitizer::BlockingMutexLock Guard(&Mutex);
  if (Buffers.empty())
    return ErrorCode::NotEnoughMemory;
  auto &T = Buffers.front();
  auto &B = std::get<0>(T);
  Buf = B;
  B.Buffer = nullptr;
  B.Size = 0;
  Buffers.pop_front();
  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::releaseBuffer(Buffer &Buf) {
  if (OwnedBuffers.count(Buf.Buffer) == 0)
    return ErrorCode::UnrecognizedBuffer;
  __sanitizer::BlockingMutexLock Guard(&Mutex);

  // Now that the buffer has been released, we mark it as "used".
  Buffers.emplace(Buffers.end(), Buf, true /* used */);
  Buf.Buffer = nullptr;
  Buf.Size = 0;
  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::finalize() {
  if (__sanitizer::atomic_exchange(&Finalizing, 1,
                                   __sanitizer::memory_order_acq_rel))
    return ErrorCode::QueueFinalizing;
  return ErrorCode::Ok;
}

BufferQueue::~BufferQueue() {
  for (auto &T : Buffers) {
    auto &Buf = std::get<0>(T);
    free(Buf.Buffer);
  }
}
