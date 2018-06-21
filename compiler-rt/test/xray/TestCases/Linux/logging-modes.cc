// Check that we can install an implementation associated with a mode.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: %run %t | FileCheck %s
//
// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include "xray/xray_interface.h"
#include "xray/xray_log_interface.h"
#include <cassert>
#include <cstdio>

[[clang::xray_never_instrument]] void printing_handler(int32_t fid,
                                                       XRayEntryType) {
  thread_local volatile bool printing = false;
  if (printing)
    return;
  printing = true;
  std::printf("printing %d\n", fid);
  printing = false;
}

[[clang::xray_never_instrument]] XRayLogInitStatus
printing_init(size_t, size_t, void *, size_t) {
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

[[clang::xray_never_instrument]] XRayLogInitStatus printing_finalize() {
  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

[[clang::xray_never_instrument]] XRayLogFlushStatus printing_flush_log() {
  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

[[clang::xray_always_instrument]] void callme() { std::printf("called me!\n"); }

static bool unused = [] {
  assert(__xray_log_register_mode("custom",
                                  {printing_init, printing_finalize,
                                   printing_handler, printing_flush_log}) ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  return true;
}();

int main(int argc, char **argv) {
  assert(__xray_log_select_mode("custom") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  assert(__xray_patch() == XRayPatchingStatus::SUCCESS);
  assert(__xray_log_init(0, 0, nullptr, 0) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);
  // CHECK: printing {{.*}}
  callme(); // CHECK: called me!
  // CHECK: printing {{.*}}
  assert(__xray_log_finalize() == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  assert(__xray_log_flushLog() == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  assert(__xray_log_select_mode("not-found") ==
         XRayLogRegisterStatus::XRAY_MODE_NOT_FOUND);
}
