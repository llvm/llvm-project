// RUN: rm -rf %t && mkdir %t
// RUN: %clangxx_xray -g -std=c++11 %s -o %t.exe
// RUN: XRAY_OPTIONS="patch_premain=false \
// RUN:    xray_logfile_base=%t/ xray_mode=xray-fdr verbosity=1" \
// RUN:    XRAY_FDR_OPTIONS=func_duration_threshold_us=0 %run %t.exe 2>&1 | \
// RUN:    FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t.exe %t/*
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t.exe %t/* | \
// RUN:   FileCheck %s --check-prefix TRACE
// FIXME: Make llvm-xray work on non-x86_64 as well.
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree

#include "xray/xray_log_interface.h"
#include <atomic>
#include <cassert>
#include <thread>

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
  __xray_patch();
  assert(__xray_log_init_mode("xray-fdr", "") ==
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
// TRACE-DAG: - { type: 0, func-id: [[FID1:[0-9]+]], function: {{.*f1.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], process: [[PROCESS:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID2:[0-9]+]], function: {{.*f2.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], process: [[PROCESS]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID1]], function: {{.*f1.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS]], kind: {{function-exit|function-tail-exit}}, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID2]], function: {{.*f2.*}}, cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: {{function-exit|function-tail-exit}}, tsc: {{[0-9]+}} }
