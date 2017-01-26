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
#include <cassert>
#include <cstdlib>

using namespace __xray;

BufferQueue::BufferQueue(std::size_t B, std::size_t N, bool &Success)
    : BufferSize(B), Buffers(N), Mutex(), OwnedBuffers(), Finalizing(false) {
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

std::error_code BufferQueue::getBuffer(Buffer &Buf) {
  if (Finalizing.load(std::memory_order_acquire))
    return std::make_error_code(std::errc::state_not_recoverable);
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Buffers.empty())
    return std::make_error_code(std::errc::not_enough_memory);
  auto &T = Buffers.front();
  auto &B = std::get<0>(T);
  Buf = B;
  B.Buffer = nullptr;
  B.Size = 0;
  Buffers.pop_front();
  return {};
}

std::error_code BufferQueue::releaseBuffer(Buffer &Buf) {
  if (OwnedBuffers.count(Buf.Buffer) == 0)
    return std::make_error_code(std::errc::argument_out_of_domain);
  std::lock_guard<std::mutex> Guard(Mutex);

  // Now that the buffer has been released, we mark it as "used".
  Buffers.emplace(Buffers.end(), Buf, true /* used */);
  Buf.Buffer = nullptr;
  Buf.Size = 0;
  return {};
}

std::error_code BufferQueue::finalize() {
  if (Finalizing.exchange(true, std::memory_order_acq_rel))
    return std::make_error_code(std::errc::state_not_recoverable);
  return {};
}

BufferQueue::~BufferQueue() {
  for (auto &T : Buffers) {
    auto &Buf = std::get<0>(T);
    free(Buf.Buffer);
  }
}
