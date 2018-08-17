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
#include "sanitizer_common/sanitizer_posix.h"
#include <memory>
#include <sys/mman.h>

using namespace __xray;
using namespace __sanitizer;

template <class T> static T *allocRaw(size_t N) {
  // TODO: Report errors?
  void *A = reinterpret_cast<void *>(
      internal_mmap(NULL, N * sizeof(T), PROT_WRITE | PROT_READ,
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
  return (A == MAP_FAILED) ? nullptr : reinterpret_cast<T *>(A);
}

template <class T> static void deallocRaw(T *ptr, size_t N) {
  // TODO: Report errors?
  if (ptr != nullptr)
    internal_munmap(ptr, N);
}

template <class T> static T *initArray(size_t N) {
  auto A = allocRaw<T>(N);
  if (A != nullptr)
    while (N > 0)
      new (A + (--N)) T();
  return A;
}

BufferQueue::BufferQueue(size_t B, size_t N, bool &Success)
    : BufferSize(B), Buffers(initArray<BufferQueue::BufferRep>(N)),
      BufferCount(N), Finalizing{0}, OwnedBuffers(initArray<void *>(N)),
      Next(Buffers), First(Buffers), LiveBuffers(0) {
  if (Buffers == nullptr) {
    Success = false;
    return;
  }
  if (OwnedBuffers == nullptr) {
    // Clean up the buffers we've already allocated.
    for (auto B = Buffers, E = Buffers + BufferCount; B != E; ++B)
      B->~BufferRep();
    deallocRaw(Buffers, N);
    Success = false;
    return;
  };

  for (size_t i = 0; i < N; ++i) {
    auto &T = Buffers[i];
    void *Tmp = allocRaw<char>(BufferSize);
    if (Tmp == nullptr) {
      Success = false;
      return;
    }
    auto *Extents = allocRaw<BufferExtents>(1);
    if (Extents == nullptr) {
      Success = false;
      return;
    }
    auto &Buf = T.Buff;
    Buf.Data = Tmp;
    Buf.Size = B;
    Buf.Extents = Extents;
    OwnedBuffers[i] = Tmp;
  }
  Success = true;
}

BufferQueue::ErrorCode BufferQueue::getBuffer(Buffer &Buf) {
  if (atomic_load(&Finalizing, memory_order_acquire))
    return ErrorCode::QueueFinalizing;
  SpinMutexLock Guard(&Mutex);
  if (LiveBuffers == BufferCount)
    return ErrorCode::NotEnoughMemory;

  auto &T = *Next;
  auto &B = T.Buff;
  Buf = B;
  T.Used = true;
  ++LiveBuffers;

  if (++Next == (Buffers + BufferCount))
    Next = Buffers;

  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::releaseBuffer(Buffer &Buf) {
  // Blitz through the buffers array to find the buffer.
  bool Found = false;
  for (auto I = OwnedBuffers, E = OwnedBuffers + BufferCount; I != E; ++I) {
    if (*I == Buf.Data) {
      Found = true;
      break;
    }
  }
  if (!Found)
    return ErrorCode::UnrecognizedBuffer;

  SpinMutexLock Guard(&Mutex);

  // This points to a semantic bug, we really ought to not be releasing more
  // buffers than we actually get.
  if (LiveBuffers == 0)
    return ErrorCode::NotEnoughMemory;

  // Now that the buffer has been released, we mark it as "used".
  First->Buff = Buf;
  First->Used = true;
  Buf.Data = nullptr;
  Buf.Size = 0;
  --LiveBuffers;
  if (++First == (Buffers + BufferCount))
    First = Buffers;

  return ErrorCode::Ok;
}

BufferQueue::ErrorCode BufferQueue::finalize() {
  if (atomic_exchange(&Finalizing, 1, memory_order_acq_rel))
    return ErrorCode::QueueFinalizing;
  return ErrorCode::Ok;
}

BufferQueue::~BufferQueue() {
  for (auto I = Buffers, E = Buffers + BufferCount; I != E; ++I) {
    auto &T = *I;
    auto &Buf = T.Buff;
    deallocRaw(Buf.Data, Buf.Size);
    deallocRaw(Buf.Extents, 1);
  }
  for (auto B = Buffers, E = Buffers + BufferCount; B != E; ++B)
    B->~BufferRep();
  deallocRaw(Buffers, BufferCount);
  deallocRaw(OwnedBuffers, BufferCount);
}
