// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm fdr-thread-order.* || true
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false \
// RUN:    xray_logfile_base=fdr-thread-order. xray_fdr_log=true verbosity=1 \
// RUN:    xray_fdr_log_func_duration_threshold_us=0" %run %t 2>&1 | \
// RUN:    FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:    "`ls fdr-thread-order.* | head -1`"
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:    "`ls fdr-thread-order.* | head -1`" | \
// RUN:    FileCheck %s --check-prefix TRACE
// RUN: rm fdr-thread-order.*
// FIXME: Make llvm-xray work on non-x86_64 as well.
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree

#include "xray/xray_log_interface.h"
#include <atomic>
#include <cassert>
#include <thread>

constexpr auto kBufferSize = 16384;
constexpr auto kBufferMax = 10;

std::atomic<uint64_t> var{0};

[[clang::xray_always_instrument]] void __attribute__((noinline)) f1() {
  for (auto i = 0; i < 1 << 20; ++i)
    ++var;
}

[[clang::xray_always_instrument]] void __attribute__((noinline)) f2() {
  for (auto i = 0; i < 1 << 20; ++i)
    ++var;
}

int main(int argc, char *argv[]) {
  using namespace __xray;
  FDRLoggingOptions Options;
  __xray_patch();
  assert(__xray_log_init(kBufferSize, kBufferMax, &Options,
                         sizeof(FDRLoggingOptions)) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);

  std::atomic_thread_fence(std::memory_order_acq_rel);

  {
    std::thread t1([] { f1(); });
    std::thread t2([] { f2(); });
    t1.join();
    t2.join();
  }

  std::atomic_thread_fence(std::memory_order_acq_rel);
  __xray_log_finalize();
  __xray_log_flushLog();
  __xray_unpatch();
  return var > 0 ? 0 : 1;
  // CHECK: {{.*}}XRay: Log file in '{{.*}}'
  // CHECK-NOT: Failed
}

// We want to make sure that the order of the function log doesn't matter.
// TRACE-DAG: - { type: 0, func-id: [[FID1:[0-9]+]], function: {{.*f1.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID2:[0-9]+]], function: {{.*f2.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID1]], function: {{.*f1.*}}, cpu: {{.*}}, thread: [[THREAD1]], kind: {{function-exit|function-tail-exit}}, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID2]], function: {{.*f2.*}}, cpu: {{.*}}, thread: [[THREAD2]], kind: {{function-exit|function-tail-exit}}, tsc: {{[0-9]+}} }
