//===-- copyprof_report_sink_test.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains unit tests for the CopyProf background report sink,
/// testing asynchronous flushing and backend interactions.
///
//===----------------------------------------------------------------------===//

#include "copyprof/copyprof_report_sink.h"

#include <new>

#include "copyprof/copyprof_interface_internal.h"
#include "copyprof/copyprof_report_backend.h"
#include "copyprof/copyprof_report_buffer.h"
#include "copyprof/copyprof_reporting.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_posix.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __copyprof {
namespace {

using ::testing::UnorderedElementsAreArray;

class CopyProfReportSinkTest : public testing::Test {
 protected:
  void SetUp() override {
    // Subtle: the CopyProf compiler-rt will have initialized reporting already
    // so stop it first as test cases will themselves initialize reporting.
    // Stopping multiple times is a noop.
    DisableBackgroundFlushing();
  }
};

// Records which buffers were flushed and how many times `Finalize` was called.
// Owned by the test because the sink frees the backend when it is disabled.
struct BackendLog {
  std::vector<ReportBuffer*> flushed_buffers;
  int num_finalize_calls = 0;
};

// A test backend that records its calls in a `BackendLog`, if one is given.
class TestBackend : public ReportBackend {
 public:
  static TestBackend* Create(BackendLog* log = nullptr) {
    return new (InternalAlloc(sizeof(TestBackend))) TestBackend(log);
  }

  void Flush(ReportBuffer* buffer) override {
    if (log_ != nullptr) {
      log_->flushed_buffers.push_back(buffer);
    }
  }
  void Finalize() override {
    if (log_ != nullptr) {
      ++log_->num_finalize_calls;
    }
  }

 private:
  explicit TestBackend(BackendLog* log) : log_(log) {}

  BackendLog* log_;
};

TEST_F(CopyProfReportSinkTest, AcquireAndReturn) {
  EnableBackgroundFlushing(TestBackend::Create());
  ReportBuffer* buffer = AcquireReportBuffer();
  EXPECT_NE(buffer, nullptr);
  ReturnReportBuffer(buffer);
  DisableBackgroundFlushing();
}

TEST_F(CopyProfReportSinkTest, ReturningBufferAfterStoppingWorks) {
  EnableBackgroundFlushing(TestBackend::Create());
  ReportBuffer* buffer = AcquireReportBuffer();
  EXPECT_NE(buffer, nullptr);
  DisableBackgroundFlushing();
  ReturnReportBuffer(buffer);
}

TEST_F(CopyProfReportSinkTest, CallbacksInvokedForAllReports) {
  // Acquires a bunch of report buffers, returns them to the sink and asserts
  // that the flush callback was invoked for each of them.
  BackendLog log;
  EnableBackgroundFlushing(TestBackend::Create(&log));
  std::vector<ReportBuffer*> acquired_buffers;
  for (int i = 0; i < 32; ++i) {
    acquired_buffers.push_back(AcquireReportBuffer());
    EXPECT_NE(acquired_buffers[i], nullptr);
  }
  for (ReportBuffer* buffer : acquired_buffers) {
    ReturnReportBuffer(buffer);
  }
  DisableBackgroundFlushing();

  EXPECT_THAT(acquired_buffers, UnorderedElementsAreArray(log.flushed_buffers));
  EXPECT_EQ(log.num_finalize_calls, 1);
}

TEST_F(CopyProfReportSinkTest, MultiThreadedStressTest) {
  // Asserts that all returned buffers are processed by the flush callback in a
  // multi-threaded program.
  // Spawns threads that each repeatedly acquire a buffer, simulate work by
  // sleeping for a short period of time, then return the buffer.
  BackendLog log;
  EnableBackgroundFlushing(TestBackend::Create(&log));

  struct ThreadArgs {
    std::vector<ReportBuffer*>* acquired_buffers;
    Mutex* acquired_buffers_mutex;
  };

  std::vector<ReportBuffer*> acquired_buffers;
  Mutex acquired_buffers_mutex;
  ThreadArgs args{&acquired_buffers, &acquired_buffers_mutex};

  auto thread_func = +[](void* arg) -> void* {
    auto* args = static_cast<ThreadArgs*>(arg);
    std::vector<ReportBuffer*> buffers;
    for (int i = 0; i < 128; ++i) {
      ReportBuffer* buffer = AcquireReportBuffer();
      buffers.push_back(buffer);
      usleep(1000);
      ReturnReportBuffer(buffer);
    }
    GenericScopedLock<Mutex> lock(args->acquired_buffers_mutex);
    args->acquired_buffers->insert(args->acquired_buffers->end(),
                                   buffers.begin(), buffers.end());
    return nullptr;
  };

  // Use half of the CPUs so that the test doesn't run for too long, but at
  // least two threads so that buffers are acquired and returned concurrently.
  u32 num_threads = Max<u32>(GetNumberOfCPUs() / 2, 2);
  std::vector<pthread_t> threads(num_threads);
  for (pthread_t& thread : threads) {
    ASSERT_EQ(pthread_create(&thread, /*attr=*/nullptr, thread_func, &args), 0);
  }
  for (pthread_t thread : threads) {
    ASSERT_EQ(pthread_join(thread, /*retval=*/nullptr), 0);
  }
  DisableBackgroundFlushing();

  EXPECT_THAT(acquired_buffers, UnorderedElementsAreArray(log.flushed_buffers));
  EXPECT_EQ(log.num_finalize_calls, 1);
}

TEST_F(CopyProfReportSinkTest, AcquireReturnsNullWhileDisabled) {
  // `SetUp` disabled background flushing.
  EXPECT_EQ(AcquireReportBuffer(), nullptr);
  ReturnReportBuffer(nullptr);  // Must be a no-op.

  EnableBackgroundFlushing(TestBackend::Create());
  ReportBuffer* buffer = AcquireReportBuffer();
  EXPECT_NE(buffer, nullptr);
  ReturnReportBuffer(buffer);
  DisableBackgroundFlushing();

  EXPECT_EQ(AcquireReportBuffer(), nullptr);
}

TEST_F(CopyProfReportSinkTest, BuffersReturnedWhileDisabledAreReused) {
  EnableBackgroundFlushing(TestBackend::Create());
  ReportBuffer* buffer = AcquireReportBuffer();
  ASSERT_NE(buffer, nullptr);
  ASSERT_NE(buffer->TryAllocate(1), nullptr);
  DisableBackgroundFlushing();
  // Returned too late to be flushed, but recycled instead of leaked.
  ReturnReportBuffer(buffer);

  EnableBackgroundFlushing(TestBackend::Create());
  EXPECT_EQ(AcquireReportBuffer(), buffer);
  EXPECT_EQ(buffer->pos(), 0u);
  ReturnReportBuffer(buffer);
  DisableBackgroundFlushing();
}

TEST_F(CopyProfReportSinkTest, LoggingAfterDisablingDropsReports) {
  // Threads may keep running after background flushing was disabled at process
  // exit. Logging reports on such threads must not crash.
  EnableBackgroundFlushing(TestBackend::Create());
  DisableBackgroundFlushing();

  auto thread_func = +[](void*) -> void* {
    GET_CURRENT_PC_BP;
    LogCopyProfReport(pc, bp, /*obj_size=*/1024, /*did_allocate=*/true);
    return nullptr;
  };
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, /*attr=*/nullptr, thread_func, nullptr), 0);
  ASSERT_EQ(pthread_join(thread, /*retval=*/nullptr), 0);
}

}  // namespace
}  // namespace __copyprof
