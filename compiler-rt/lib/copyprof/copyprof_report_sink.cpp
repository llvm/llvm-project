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
  pthread_t handle_;
  bool is_valid_ = false;
};

// Used for allocating new buffers and for returning full buffers. The
// background thread flushes reports periodically to move printing of reports
// out of the critical path. `AcquireReportBuffer` and `ReturnReportBuffer` are
// the only thread-safe methods. All other methods may only be called from the
// main thread (typically at startup or when the application exits).
class ReportSink {
 public:
  // Takes ownership of `backend`.
  ReportSink(ReportBackend* backend);

  // Acquires a fresh buffer to store CopyProf reports to.
  // Must be returned before program exit, otherwise it won't be part of the
  // CopyProf output.
  ReportBuffer* AcquireReportBuffer();
  void ReturnReportBuffer(ReportBuffer* buffer);

  // Creates and starts a background thread that flushes pending reports
  // periodically.
  void Activate();
  // Flushes pending reports if any, and blocks until the background thread has
  // exited. Then invokes the close callback if one was set.
  void FlushAndDeactivate();
  // Whether background report flushing is active.
  bool IsActive();

 private:
  friend void* ThreadStart(void* arg);

  ReportBuffer* GetBufferLocked() SANITIZER_REQUIRES(mutex_);
  ReportBuffer* GetBuffer() SANITIZER_EXCLUDES(mutex_);
  ReportBuffer* GetNextPendingBuffer() SANITIZER_EXCLUDES(mutex_);
  void BackgroundThread(ReportBackend& backend);

  // Used to lock access to the buffer lists.
  Mutex mutex_;
  // Contains `ReportBuffer`s available to user threads.
  alignas(SANITIZER_CACHE_LINE_SIZE)
      IntrusiveList<ReportBuffer> available_buffers_
      SANITIZER_GUARDED_BY(mutex_);
  // Contains `ReportBuffer`s that have been returned but not flushed yet.
  alignas(SANITIZER_CACHE_LINE_SIZE)
      IntrusiveList<ReportBuffer> pending_flush_buffers_
      SANITIZER_GUARDED_BY(mutex_);

  PthreadHandle thread_handle_;
  Semaphore flush_semaphore_;
  atomic_uint8_t stop_background_thread_;
  ReportBackend* backend_;
};

ReportSink* g_sink = nullptr;

ReportSink::ReportSink(ReportBackend* backend) : backend_(backend) {
  CHECK(backend != nullptr);
  IntrusiveList<ReportBuffer> buffers = AllocateBuffers();
  available_buffers_.append_front(&buffers);
}

// Returns an available buffer or `nullptr` if none exists.
ReportBuffer* ReportSink::GetBufferLocked() {
  if (available_buffers_.empty()) {
    return nullptr;
  }
  ReportBuffer* buffer = available_buffers_.front();
  available_buffers_.pop_front();
  return buffer;
}

ReportBuffer* ReportSink::GetBuffer() {
  GenericScopedLock<Mutex> lock(&mutex_);
  return GetBufferLocked();
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

void ReportSink::Activate() {
  atomic_store(&stop_background_thread_, 0, memory_order_release);
  pthread_t handle;
  auto* param = static_cast<ThreadParam*>(InternalAlloc(sizeof(ThreadParam)));
  param->sink = this;
  param->backend = backend_;
  CHECK(pthread_create(&handle, nullptr, ThreadStart, param) == 0);
  thread_handle_ = PthreadHandle(handle);
}

void ReportSink::FlushAndDeactivate() {
  if (!thread_handle_.IsValid()) {
    return;
  }
  atomic_store(&stop_background_thread_, 1, memory_order_release);
  flush_semaphore_.Post();
  pthread_join(thread_handle_.Handle(), /*__thread_return=*/nullptr);
  thread_handle_ = {};
  backend_->Finalize();
  InternalFree(backend_);
  backend_ = nullptr;
}

bool ReportSink::IsActive() { return thread_handle_.IsValid(); }

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
  if (ReportBuffer* buffer = GetBuffer(); buffer != nullptr) {
    return buffer;
  }
  // Slow path: allocate new buffers but avoid holding the lock while mapping
  // memory.
  IntrusiveList<ReportBuffer> new_buffers = AllocateBuffers();
  // Note that it's not a problem if locking here races with the main thread
  // exiting since reports that are generated by background threads while
  // shutdown is running will be missed anyways.
  GenericScopedLock<Mutex> lock(&mutex_);
  available_buffers_.append_front(&new_buffers);
  return GetBufferLocked();
}

void ReportSink::ReturnReportBuffer(ReportBuffer* buffer) {
  GenericScopedLock<Mutex> lock(&mutex_);
  pending_flush_buffers_.push_front(buffer);
  flush_semaphore_.Post();
}

}  // namespace

ReportBuffer* AcquireReportBuffer() { return g_sink->AcquireReportBuffer(); }

void ReturnReportBuffer(ReportBuffer* buffer) {
  if (g_sink != nullptr) {
    g_sink->ReturnReportBuffer(buffer);
  }
}

void DisableBackgroundFlushing() {
  if (g_sink != nullptr) {
    g_sink->FlushAndDeactivate();
    g_sink = nullptr;
  }
}

void EnableBackgroundFlushing(ReportBackend* backend) {
  CHECK(g_sink == nullptr || !g_sink->IsActive());
  if (g_sink == nullptr) {
    static char sink_storage[sizeof(ReportSink)];
    g_sink = new (sink_storage) ReportSink(backend);
  }
  g_sink->Activate();
}

}  // namespace __copyprof
