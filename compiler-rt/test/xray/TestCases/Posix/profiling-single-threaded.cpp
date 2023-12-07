// Check that we can get a profile from a single-threaded application, on
// demand through the XRay logging implementation API.
//
// FIXME: Make -fxray-modes=xray-profiling part of the default?
// RUN: %clangxx_xray -std=c++11 %s -o %t -fxray-modes=xray-profiling
// RUN: rm -f xray-log.profiling-single-*
// RUN: XRAY_OPTIONS=verbosity=1 \
// RUN:     XRAY_PROFILING_OPTIONS=no_flush=true %run %t
// RUN: XRAY_OPTIONS=verbosity=1 %run %t
// RUN: PROFILES=`ls xray-log.profiling-single-* | wc -l`
// RUN: [ $PROFILES -eq 2 ]
// RUN: rm -f xray-log.profiling-single-*
//
// REQUIRES: built-in-llvm-tree

#include "xray/xray_interface.h"
#include "xray/xray_log_interface.h"
#include <cassert>
#include <cstdio>
#include <string>

[[clang::xray_always_instrument]] void f2() { return; }
[[clang::xray_always_instrument]] void f1() { f2(); }
[[clang::xray_always_instrument]] void f0() { f1(); }

using namespace std;

volatile int buffer_counter = 0;

[[clang::xray_never_instrument]] void process_buffer(const char *, XRayBuffer) {
  // FIXME: Actually assert the contents of the buffer.
  ++buffer_counter;
}

[[clang::xray_always_instrument]] int main(int, char **) {
  assert(__xray_log_select_mode("xray-profiling") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  assert(__xray_log_get_current_mode() != nullptr);
  std::string current_mode = __xray_log_get_current_mode();
  assert(current_mode == "xray-profiling");
  assert(__xray_patch() == XRayPatchingStatus::SUCCESS);
  assert(__xray_log_init_mode("xray-profiling", "") ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  f0();
  assert(__xray_log_finalize() == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  f0();
  assert(__xray_log_process_buffers(process_buffer) ==
         XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  // There's always at least one buffer, containing the profile file header. We
  // assert that we have two, to indicate that we're expecting exactly one
  // thread's worth of data.
  assert(buffer_counter == 2);
  assert(__xray_log_flushLog() == XRayLogFlushStatus::XRAY_LOG_FLUSHED);

  // Let's reset the counter.
  buffer_counter = 0;

  assert(__xray_log_init_mode("xray-profiling", "") ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  f0();
  assert(__xray_log_finalize() == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  f0();
  assert(__xray_log_process_buffers(process_buffer) ==
         XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  assert(buffer_counter == 2);
  assert(__xray_log_flushLog() == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
}
