// RUN: %clangxx_xray -g -std=c++11 %s -o %t -fxray-modes=xray-fdr
// RUN: rm -f fdr-inmemory-test-*
// RUN: XRAY_OPTIONS="patch_premain=false xray_logfile_base=fdr-inmemory-test- \
// RUN:     verbosity=1" \
// RUN: XRAY_FDR_OPTIONS="no_file_flush=true func_duration_threshold_us=0" \
// RUN:     %run %t 2>&1 | FileCheck %s
// RUN: FILES=`find %T -name 'fdr-inmemory-test-*' | wc -l`
// RUN: [ $FILES -eq 0 ]
// RUN: rm -f fdr-inmemory-test-*
//
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree

#include "xray/xray_log_interface.h"
#include <cassert>
#include <iostream>

uint64_t var = 0;
uint64_t buffers = 0;
[[clang::xray_always_instrument]] void __attribute__((noinline)) f() { ++var; }

int main(int argc, char *argv[]) {
  assert(__xray_log_select_mode("xray-fdr") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  auto status = __xray_log_init_mode(
      "xray-fdr",
      "buffer_size=4096:buffer_max=10:func_duration_threshold_us=0");
  assert(status == XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  __xray_patch();

  // Create enough entries.
  for (int i = 0; i != 1 << 20; ++i) {
    f();
  }

  // Then we want to verify that we're getting 10 buffers outside of the initial
  // header.
  auto finalize_status = __xray_log_finalize();
  assert(finalize_status == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  auto process_status =
      __xray_log_process_buffers([](const char *, XRayBuffer) { ++buffers; });
  std::cout << "buffers = " << buffers << std::endl;
  assert(process_status == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  auto flush_status = __xray_log_flushLog();
  assert(flush_status == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  // We expect 11 buffers because 1 header buffer + 10 actual FDR buffers.
  // CHECK: Buffers = 11
  std::cout << "Buffers = " << buffers << std::endl;

  // In this test we ensure that we can restart the cycle after the flush.
  status = __xray_log_init_mode(
      "xray-fdr",
      "buffer_size=4096:buffer_max=10:func_duration_threshold_us=0");
  assert(status == XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  __xray_patch();

  // Create enough entries.
  for (int i = 0; i != 1 << 20; ++i) {
    f();
  }

  // Then we want to verify that we're getting 10 buffers outside of the initial
  // header.
  finalize_status = __xray_log_finalize();
  assert(finalize_status == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  process_status =
      __xray_log_process_buffers([](const char *, XRayBuffer) { ++buffers; });
  std::cout << "buffers = " << buffers << std::endl;
  assert(process_status == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  flush_status = __xray_log_flushLog();
  assert(flush_status == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  // We expect 22 buffers because 1 header buffer + 10 actual FDR buffers, plus
  // the number of buffers we got from the previous run (also 11).
  // CHECK: Buffers = 22
  std::cout << "Buffers = " << buffers << std::endl;
}
