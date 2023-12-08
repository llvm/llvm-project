// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm -f fdr-logging-test-*
// RUN: rm -f fdr-unwrite-test-*
// RUN: XRAY_OPTIONS="patch_premain=false xray_logfile_base=fdr-logging-test- \
// RUN:     xray_mode=xray-fdr verbosity=1" \
// RUN: XRAY_FDR_OPTIONS="func_duration_threshold_us=0" \
// RUN:     %run %t 2>&1 | FileCheck %s
// RUN: XRAY_OPTIONS="patch_premain=false \
// RUN:     xray_logfile_base=fdr-unwrite-test- xray_mode=xray-fdr \
// RUN:     verbosity=1" \
// RUN: XRAY_FDR_OPTIONS="func_duration_threshold_us=5000" \
// RUN:     %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:     "`ls fdr-logging-test-* | head -1`" \
// RUN:     | FileCheck %s --check-prefix=TRACE
// RUN: %llvm_xray convert --symbolize --output-format=yaml -instr_map=%t \
// RUN:     "`ls fdr-unwrite-test-* | head -1`" \
// RUN:     | FileCheck %s --check-prefix=UNWRITE
// RUN: rm fdr-logging-test-*
// RUN: rm fdr-unwrite-test-*
// UNSUPPORTED: target=powerpc64le-{{.*}}
/// TODO: FDR logging arg1 handler(__xray_ArgLoggerEntry) hasn't implemented yet on LoongArch
// UNSUPPORTED: target=loongarch64{{.*}}
// REQUIRES: built-in-llvm-tree

#include "xray/xray_log_interface.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <thread>
#include <time.h>

thread_local uint64_t var = 0;
[[clang::xray_always_instrument]] void __attribute__((noinline)) fC() { ++var; }

[[clang::xray_always_instrument]] void __attribute__((noinline)) fB() { fC(); }

[[clang::xray_always_instrument]] void __attribute__((noinline)) fA() { fB(); }

[[clang::xray_always_instrument, clang::xray_log_args(1)]]
void __attribute__((noinline)) fArg(int) { }

int main(int argc, char *argv[]) {
  std::cout << "Logging before init." << std::endl;
  // CHECK: Logging before init.
  assert(__xray_log_select_mode("xray-fdr") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  auto status =
      __xray_log_init_mode("xray-fdr", "buffer_size=16384:buffer_max=10");
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
    fArg(1);
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
// TRACE-DAG: - { type: 0, func-id: [[FIDA:[0-9]+]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], process: [[PROCESS:[0-9]+]], kind: function-enter, tsc: {{[0-9]+}}, data: '' }
// TRACE:     - { type: 0, func-id: [[FIDA]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}}, data: '' }
// TRACE-DAG: - { type: 0, func-id: [[FIDA]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], process: [[PROCESS]], kind: function-enter, tsc: {{[0-9]+}}, data: '' }
// TRACE:     - { type: 0, func-id: [[FIDA]], function: {{.*fA.*}}, cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}}, data: '' }
//
// Do the same as above for fC()
// TRACE-DAG: - { type: 0, func-id: [[FIDC:[0-9]+]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], process: [[PROCESS]], kind: function-enter, tsc: {{[0-9]+}}, data: '' }
// TRACE:     - { type: 0, func-id: [[FIDC]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}}, data: '' }
// TRACE-DAG: - { type: 0, func-id: [[FIDC]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], process: [[PROCESS]], kind: function-enter, tsc: {{[0-9]+}}, data: '' }
// TRACE:     - { type: 0, func-id: [[FIDC]], function: {{.*fC.*}}, cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}}, data: '' }

// Do the same as above for fB()
// TRACE-DAG: - { type: 0, func-id: [[FIDB:[0-9]+]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD1:[0-9]+]], process: [[PROCESS]], kind: function-enter, tsc: {{[0-9]+}}, data: '' }
// TRACE:     - { type: 0, func-id: [[FIDB]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD1]], process: [[PROCESS]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}}, data: '' }
// TRACE-DAG: - { type: 0, func-id: [[FIDB]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], process: [[PROCESS]], kind: function-enter, tsc: {{[0-9]+}}, data: '' }
// TRACE:     - { type: 0, func-id: [[FIDB]], function: {{.*fB.*}}, cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: function-{{exit|tail-exit}}, tsc: {{[0-9]+}}, data: '' }

// TRACE-DAG: - { type: 0, func-id: [[FIDARG:[0-9]+]], function: 'fArg(int)', args: [ 1 ], cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: function-enter-arg, tsc: {{[0-9]+}}, data: '' }
// TRACE-DAG: - { type: 0, func-id: [[FIDARG]], function: 'fArg(int)', cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: function-exit, tsc: {{[0-9]+}}, data: '' }

// Assert that when unwriting is enabled with a high threshold time, all the function records are erased. A CPU switch could erroneously fail this test, but
// is unlikely given the test program.
// Even with a high threshold, arg1 logging is never unwritten.
// UNWRITE: header:
// UNWRITE: records:
// UNWRITE-NEXT: - { type: 0, func-id: [[FIDARG:[0-9]+]], function: 'fArg(int)', args: [ 1 ], cpu: {{.*}}, thread: [[THREAD2:[0-9]+]], process: [[PROCESS:[0-9]+]], kind: function-enter-arg, tsc: {{[0-9]+}}, data: '' }
// UNWRITE-NEXT: - { type: 0, func-id: [[FIDARG]], function: 'fArg(int)', cpu: {{.*}}, thread: [[THREAD2]], process: [[PROCESS]], kind: function-exit, tsc: {{[0-9]+}}, data: '' }
// UNWRITE-NOT: function-enter
// UNWRITE-NOT: function-{{exit|tail-exit}}
