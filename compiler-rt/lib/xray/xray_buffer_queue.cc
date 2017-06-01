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

BufferQueue::BufferQueue(std::size_t B, std::size_t N)
    : BufferSize(B), Buffers(N), Mutex(), OwnedBuffers(), Finalizing(false) {
  for (auto &Buf : Buffers) {
    void *Tmp = malloc(BufferSize);
    Buf.Buffer = Tmp;
    Buf.Size = B;
    if (Tmp != 0)
      OwnedBuffers.insert(Tmp);
  }
}

std::error_code BufferQueue::getBuffer(Buffer &Buf) {
  if (Finalizing.load(std::memory_order_acquire))
    return std::make_error_code(std::errc::state_not_recoverable);
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Buffers.empty())
    return std::make_error_code(std::errc::not_enough_memory);
  Buf = Buffers.front();
  Buffers.pop_front();
  return {};
}

std::error_code BufferQueue::releaseBuffer(Buffer &Buf) {
  if (OwnedBuffers.count(Buf.Buffer) == 0)
    return std::make_error_code(std::errc::argument_out_of_domain);
  std::lock_guard<std::mutex> Guard(Mutex);
  Buffers.push_back(Buf);
  Buf.Buffer = nullptr;
  Buf.Size = BufferSize;
  return {};
}

std::error_code BufferQueue::finalize() {
  if (Finalizing.exchange(true, std::memory_order_acq_rel))
    return std::make_error_code(std::errc::state_not_recoverable);
  return {};
}

BufferQueue::~BufferQueue() {
  for (auto &Buf : Buffers) {
    free(Buf.Buffer);
    Buf.Buffer = nullptr;
    Buf.Size = 0;
  }
}
