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
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_posix.h"

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

// A test backend that records which buffers were flushed and how many times
// Finalize was called.
class TestBackend : public ReportBackend {
 public:
  static TestBackend* Create() {
    return new (InternalAlloc(sizeof(TestBackend))) TestBackend();
  }

  void Flush(ReportBuffer* buffer) override {
    flushed_buffers.push_back(buffer);
  }
  void Finalize() override { ++num_finalize_calls; }

  std::vector<ReportBuffer*> flushed_buffers;
  int num_finalize_calls = 0;
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
  TestBackend* backend = TestBackend::Create();
  // Save pointer before passing ownership to EnableBackgroundFlushing.
  TestBackend* backend_ref = backend;
  EnableBackgroundFlushing(backend);
  std::vector<ReportBuffer*> acquired_buffers;
  for (int i = 0; i < 32; ++i) {
    acquired_buffers.push_back(AcquireReportBuffer());
    EXPECT_NE(acquired_buffers[i], nullptr);
  }
  for (ReportBuffer* buffer : acquired_buffers) {
    ReturnReportBuffer(buffer);
  }
  DisableBackgroundFlushing();

  EXPECT_THAT(acquired_buffers,
              UnorderedElementsAreArray(backend_ref->flushed_buffers));
  EXPECT_EQ(backend_ref->num_finalize_calls, 1);
}

TEST_F(CopyProfReportSinkTest, MultiThreadedStressTest) {
  // Asserts that all returned buffers are processed by the flush callback in a
  // multi-threaded program.
  // Spawns one thread for each other CPU in the system that each acquires a
  // buffer, simulates work by sleeping for a short period of time, then returns
  // the buffer.
  TestBackend* backend = TestBackend::Create();
  EnableBackgroundFlushing(backend);

  struct ThreadArgs {
    std::vector<ReportBuffer*>* acquired_buffers;
    Mutex* acquired_buffers_mutex;
  };

  std::vector<ReportBuffer*> acquired_buffers;
  Mutex acquired_buffers_mutex;
  ThreadArgs args{&acquired_buffers, &acquired_buffers_mutex};

  auto thread_func = +[](void* arg) -> void* {
    auto* args = reinterpret_cast<ThreadArgs*>(arg);
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

  // Don't spawn too many threads otherwise test runs for too long.
  int num_cpus = GetNumberOfCPUs() / 2;
  std::vector<pthread_t> threads(num_cpus);
  for (int i = 0; i < num_cpus; ++i) {
    pthread_create(&threads[i], /*attr=*/nullptr, thread_func, &args);
  }
  for (int i = 0; i < num_cpus; ++i) {
    pthread_join(threads[i], /*retval=*/nullptr);
  };
  DisableBackgroundFlushing();

  EXPECT_THAT(acquired_buffers,
              UnorderedElementsAreArray(backend->flushed_buffers));
  EXPECT_EQ(backend->num_finalize_calls, 1);
}

}  // namespace
}  // namespace __copyprof
