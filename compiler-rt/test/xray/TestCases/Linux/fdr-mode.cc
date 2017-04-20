// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm fdr-logging-test-* || true
// RUN: rm fdr-unwrite-test-* || true
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false xray_logfile_base=fdr-logging-test- xray_fdr_log=true verbosity=1 xray_fdr_log_func_duration_threshold_us=0" %run %t 2>&1 | FileCheck %s
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false xray_logfile_base=fdr-unwrite-test- xray_fdr_log=true verbosity=1 xray_fdr_log_func_duration_threshold_us=5000" %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t "`ls fdr-logging-test-* | head -1`" | FileCheck %s --check-prefix=TRACE
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t "`ls fdr-unwrite-test-* | head -1`" | FileCheck %s --check-prefix=UNWRITE
// RUN: rm fdr-logging-test-*
// RUN: rm fdr-unwrite-test-*
// FIXME: Make llvm-xray work on non-x86_64 as well.
// REQUIRES: x86_64-linux
// REQUIRES: built-in-llvm-tree

#include "xray/xray_log_interface.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <time.h>

constexpr auto kBufferSize = 16384;
constexpr auto kBufferMax = 10;

thread_local uint64_t var = 0;
[[clang::xray_always_instrument]] void __attribute__((noinline)) fC() { ++var; }

[[clang::xray_always_instrument]] void __attribute__((noinline)) fB() { fC(); }

[[clang::xray_always_instrument]] void __attribute__((noinline)) fA() { fB(); }

int main(int argc, char *argv[]) {
  using namespace __xray;
  FDRLoggingOptions Options;
  std::cout << "Logging before init." << std::endl;
  // CHECK: Logging before init.
  auto status = __xray_log_init(kBufferSize, kBufferMax, &Options,
                                sizeof(FDRLoggingOptions));
  assert(status == XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  std::cout << "Init status " << status << std::endl;
  // CHECK: Init status {{.*}}
  std::cout << "Patching..." << std::endl;
  // CHECK: Patching...
  __xray_patch();
  fA();
  fC();
  fB();
  fA();
  fC();
  std::thread other_thread([]() {
    fC();
    fB();
    fA();
  });
  other_thread.join();
  std::cout << "Joined" << std::endl;
  // CHECK: Joined
  std::cout << "Finalize status " << __xray_log_finalize() << std::endl;
  // CHECK: Finalize status {{.*}}
  fC();
  std::cout << "Main execution var = " << var << std::endl;
  // CHECK: Main execution var = 6
  std::cout << "Flush status " << __xray_log_flushLog() << std::endl;
  // CHECK: Flush status {{.*}}
  __xray_unpatch();
  return 0;
}

// Check that we're able to see two threads, each entering and exiting fA().
// TRACE-DAG: - { type: 0, func-id: [[FIDA:[0-9]+]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE:     - { type: 0, func-id: [[FIDA]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD1]], kind: function-exit, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FIDA]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE:     - { type: 0, func-id: [[FIDA]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD2]], kind: function-exit, tsc: {{[0-9]+}} }
//
// Do the same as above for fC()
// TRACE-DAG: - { type: 0, func-id: [[FIDC:[0-9]+]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE:     - { type: 0, func-id: [[FIDC]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD1]], kind: function-exit, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FIDC]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE:     - { type: 0, func-id: [[FIDC]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD2]], kind: function-exit, tsc: {{[0-9]+}} }

// Do the same as above for fB()
// TRACE-DAG: - { type: 0, func-id: [[FIDB:[0-9]+]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE:     - { type: 0, func-id: [[FIDB]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD1]], kind: function-exit, tsc: {{[0-9]+}} }
// TRACE-DAG: - { type: 0, func-id: [[FIDB]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}} }
// TRACE:     - { type: 0, func-id: [[FIDB]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD2]], kind: function-exit, tsc: {{[0-9]+}} }

// Assert that when unwriting is enabled with a high threshold time, all the function records are erased. A CPU switch could erroneously fail this test, but
// is unlikely given the test program.
// UNWRITE: header
// UNWRITE-NOT: function-enter
// UNWRITE-NOT: function-exit
