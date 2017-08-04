// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm fdr-thread-order.* || true
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false xray_logfile_base=fdr-thread-order. xray_fdr_log=true verbosity=1 xray_fdr_log_func_duration_threshold_us=0" %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t "`ls fdr-thread-order.* | head -1`" | FileCheck %s --check-prefix TRACE
// RUN: rm fdr-thread-order.*
// FIXME: Make llvm-xray work on non-x86_64 as well.
// REQUIRES: x86_64-linux
// REQUIRES: built-in-llvm-tree

#include "xray/xray_log_interface.h"
#include <thread>
#include <cassert>

constexpr auto kBufferSize = 16384;
constexpr auto kBufferMax = 10;

thread_local uint64_t var = 0;
[[clang::xray_always_instrument]] void __attribute__((noinline)) f1() { ++var; }
[[clang::xray_always_instrument]] void __attribute__((noinline)) f2() { ++var; }

int main(int argc, char *argv[]) {
  using namespace __xray;
  FDRLoggingOptions Options;
  assert(__xray_log_init(kBufferSize, kBufferMax, &Options,
                         sizeof(FDRLoggingOptions)) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  __xray_patch();
  std::thread t1([] { f1(); });
  std::thread t2([] { f2(); });
  t1.join();
  t2.join();
  __xray_log_finalize();
  __xray_log_flushLog();
  // CHECK: =={{[0-9]+}}==XRay: Log file in '{{.*}}'
}

// We want to make sure that the order of the function log doesn't matter.
// TRACE-DAG: - { type: 0, func-id: [[FID1:[0-9]+]], function: {{.*f1.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID2:[0-9]+]], function: {{.*f2.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID1]], function: {{.*f1.*}}, cpu: {{.*}}, thread: [[THREAD1]], kind: function-exit, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FID2]], function: {{.*f2.*}}, cpu: {{.*}}, thread: [[THREAD2]], kind: function-exit, tsc: {{[0-9]+}} }
