// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false xray_naive_log=false xray_logfile_base=fdr-logging-test- xray_fdr_log=true verbosity=1" %run %t 2>&1 | FileCheck %s
// FIXME: %llvm_xray convert -instr_map=%t "`ls fdr-logging-test-* | head -1`" | FileCheck %s --check-prefix TRACE
// RUN: rm fdr-logging-test-*

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

// TRACE: { function }
