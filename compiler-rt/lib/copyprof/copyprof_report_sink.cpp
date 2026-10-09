//===-- copyprof_report_sink.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the background reporting sink for CopyProf, managing
/// asynchronous flushing of buffered reports to configured backends.
///
//===----------------------------------------------------------------------===//

#include "copyprof_report_sink.h"

#include <pthread.h>
#include <string.h>

#include "copyprof_per_object_state.h"
#include "copyprof_report.h"
#include "copyprof_report_backend.h"
#include "copyprof_report_buffer.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __copyprof {
namespace {

constexpr int kBufferBatchSize = 16;      // arbitrary
constexpr usize kReportsPerBuffer = 512;  // arbitrary
constexpr usize kReportBufferByteSize =
    sizeof(CopyProfReport) * kReportsPerBuffer;

IntrusiveList<ReportBuffer> AllocateBuffers() {
  IntrusiveList<ReportBuffer> buffers;
  // `IntrusiveList` has no c'tor so it must be initialized manually.
  buffers.clear();
  for (int i = 0; i < kBufferBatchSize; ++i) {
    usize size = RoundUpTo(kReportBufferByteSize, GetPageSizeCached());
    buffers.push_front(new (MmapOrDie(size, "CopyProfReportBuffer"))
                           ReportBuffer(size - sizeof(ReportBuffer)));
  }
  return buffers;
}

// There's no portable way to check whether `pthread_t` points to a valid
// thread handle so use this wrapper to express optionality.
class PthreadHandle {
 public:
  PthreadHandle() = default;
  explicit PthreadHandle(pthread_t handle) : handle_(handle), is_valid_(true) {}
  bool IsValid() const { return is_valid_; }
  pthread_t Handle() const { return handle_; }

 private:
  pthread_t handle_ = {};
  bool is_valid_ = false;
};

// Used for allocating new buffers and for returning full buffers. The
// background thread flushes reports periodically to move printing of reports
// out of the critical path. `AcquireReportBuffer` and `ReturnReportBuffer` are
// the only thread-safe methods. All other methods may only be called from the
// main thread (typically at startup or when the application exits).
//
// The only instance is `g_sink`. It is never destroyed, so threads can use it
// at any time, even before `Enable` or after `Disable`. While the sink isn't
// accepting reports, it hands out no buffers and recycles returned buffers
// without flushing them.
class ReportSink {
 public:
  // Constant-initializes the sink so that it can be used before the runtime is
  // initialized.
  constexpr ReportSink() = default;

  // Acquires a fresh buffer to store CopyProf reports to, or returns `nullptr`
  // if the sink isn't accepting reports.
  // Must be returned before program exit, otherwise it won't be part of the
  // CopyProf output.
  ReportBuffer* AcquireReportBuffer();
  // Queues `buffer` for flushing if the sink is accepting reports. Otherwise,
  // drops the buffer's reports and recycles it.
  void ReturnReportBuffer(ReportBuffer* buffer);

  // Takes ownership of `backend`, starts a background thread that flushes
  // pending reports periodically, and starts accepting reports.
  void Enable(ReportBackend* backend);
  // Stops accepting reports, flushes pending reports if any, and blocks until
  // the background thread has exited. Does nothing if the sink isn't enabled.
  void Disable();

 private:
  friend void* ThreadStart(void* arg);

  ReportBuffer* GetBufferLocked() SANITIZER_REQUIRES(mutex_);
  ReportBuffer* GetNextPendingBuffer() SANITIZER_EXCLUDES(mutex_);
  void BackgroundThread(ReportBackend& backend);

  // Used to lock access to the buffer lists and `accepting_`.
  Mutex mutex_;
  // Whether the sink hands out buffers and flushes returned ones.
  bool accepting_ SANITIZER_GUARDED_BY(mutex_) = false;
  // Contains `ReportBuffer`s available to user threads.
  alignas(SANITIZER_CACHE_LINE_SIZE)
      IntrusiveList<ReportBuffer> available_buffers_
      SANITIZER_GUARDED_BY(mutex_) = {};
  // Contains `ReportBuffer`s that have been returned but not flushed yet.
  alignas(SANITIZER_CACHE_LINE_SIZE)
      IntrusiveList<ReportBuffer> pending_flush_buffers_
      SANITIZER_GUARDED_BY(mutex_) = {};

  PthreadHandle thread_handle_;
  Semaphore flush_semaphore_;
  atomic_uint8_t stop_background_thread_ = {0};
  ReportBackend* backend_ = nullptr;
};

// Constant-initialized and trivially destructible, so it's usable from before
// the runtime is initialized until the process terminates.
ReportSink g_sink;

// Returns an available buffer or `nullptr` if none exists.
ReportBuffer* ReportSink::GetBufferLocked() {
  if (available_buffers_.empty()) {
    return nullptr;
  }
  ReportBuffer* buffer = available_buffers_.front();
  available_buffers_.pop_front();
  return buffer;
}

ReportBuffer* ReportSink::GetNextPendingBuffer() {
  GenericScopedLock<Mutex> lock(&mutex_);
  if (pending_flush_buffers_.empty()) {
    return nullptr;
  }
  ReportBuffer* buffer = pending_flush_buffers_.front();
  pending_flush_buffers_.pop_front();
  return buffer;
}

struct ThreadParam {
  ReportSink* sink;
  ReportBackend* backend;
};

void* ThreadStart(void* arg) {
  ThreadParam* param = reinterpret_cast<ThreadParam*>(arg);
  ReportSink* sink = param->sink;
  ReportBackend* backend = param->backend;
  InternalFree(param);
  sink->BackgroundThread(*backend);
  return nullptr;
}

void ReportSink::Enable(ReportBackend* backend) {
  CHECK(backend != nullptr);
  CHECK(!thread_handle_.IsValid());
  backend_ = backend;
  atomic_store(&stop_background_thread_, 0, memory_order_release);
  pthread_t handle;
  auto* param = static_cast<ThreadParam*>(InternalAlloc(sizeof(ThreadParam)));
  param->sink = this;
  param->backend = backend_;
  CHECK(pthread_create(&handle, nullptr, ThreadStart, param) == 0);
  thread_handle_ = PthreadHandle(handle);

  GenericScopedLock<Mutex> lock(&mutex_);
  // Allocate the first batch up front so that the first reports don't have to.
  if (available_buffers_.empty()) {
    IntrusiveList<ReportBuffer> buffers = AllocateBuffers();
    available_buffers_.append_front(&buffers);
  }
  accepting_ = true;
}

void ReportSink::Disable() {
  if (!thread_handle_.IsValid()) {
    return;
  }
  {
    GenericScopedLock<Mutex> lock(&mutex_);
    accepting_ = false;
  }
  atomic_store(&stop_background_thread_, 1, memory_order_release);
  flush_semaphore_.Post();
  pthread_join(thread_handle_.Handle(), /*__thread_return=*/nullptr);
  thread_handle_ = {};
  backend_->Finalize();
  InternalFree(backend_);
  backend_ = nullptr;
}

void ReportSink::BackgroundThread(ReportBackend& backend) {
  // Blocks on semaphore until other threads post work, then periodically
  // flushes pending reports.
  while (!atomic_load(&stop_background_thread_, memory_order_acquire)) {
    flush_semaphore_.Wait();
    // Subtle: might have been woken up w/o any pending buffers.
    if (ReportBuffer* buffer = GetNextPendingBuffer(); buffer != nullptr) {
      backend.Flush(buffer);
      buffer->Reset();
      GenericScopedLock<Mutex> lock(&mutex_);
      available_buffers_.push_front(buffer);
    }
  }

  // Drain remaining reports, if any.
  for (ReportBuffer* buffer = GetNextPendingBuffer(); buffer != nullptr;
       buffer = GetNextPendingBuffer()) {
    backend.Flush(buffer);
    buffer->Reset();
    GenericScopedLock<Mutex> lock(&mutex_);
    available_buffers_.push_front(buffer);
  }
}

ReportBuffer* ReportSink::AcquireReportBuffer() {
  {
    GenericScopedLock<Mutex> lock(&mutex_);
    if (!accepting_) {
      return nullptr;
    }
    if (ReportBuffer* buffer = GetBufferLocked(); buffer != nullptr) {
      return buffer;
    }
  }
  // Slow path: allocate new buffers but avoid holding the lock while mapping
  // memory.
  IntrusiveList<ReportBuffer> new_buffers = AllocateBuffers();
  GenericScopedLock<Mutex> lock(&mutex_);
  available_buffers_.append_front(&new_buffers);
  // The sink may have stopped accepting reports in the meantime.
  return accepting_ ? GetBufferLocked() : nullptr;
}

void ReportSink::ReturnReportBuffer(ReportBuffer* buffer) {
  GenericScopedLock<Mutex> lock(&mutex_);
  if (accepting_) {
    pending_flush_buffers_.push_front(buffer);
    flush_semaphore_.Post();
    return;
  }
  buffer->Reset();
  available_buffers_.push_front(buffer);
}

}  // namespace

ReportBuffer* AcquireReportBuffer() { return g_sink.AcquireReportBuffer(); }

void ReturnReportBuffer(ReportBuffer* buffer) {
  if (buffer != nullptr) {
    g_sink.ReturnReportBuffer(buffer);
  }
}

void DisableBackgroundFlushing() { g_sink.Disable(); }

void EnableBackgroundFlushing(ReportBackend* backend) {
  g_sink.Enable(backend);
}

}  // namespace __copyprof
