// Check that we can install an implementation associated with a mode.
//
// RUN: rm xray-log.logging-modes* || true
// RUN: %clangxx_xray -std=c++11 %s -o %t -fxray-modes=none
// RUN: %run %t | FileCheck %s
//
// UNSUPPORTED: target-is-mips64,target-is-mips64el

#include "xray/xray_interface.h"
#include "xray/xray_log_interface.h"
#include <cassert>
#include <cstdio>
#include <string>

[[clang::xray_never_instrument]] void printing_handler(int32_t fid,
                                                       XRayEntryType) {
  thread_local volatile bool printing = false;
  if (printing)
    return;
  printing = true;
  std::printf("printing %d\n", fid);
  printing = false;
}

[[clang::xray_never_instrument]] XRayBuffer next_buffer(XRayBuffer buffer) {
  static const char data[10] = {};
  static const XRayBuffer first_and_last{data, 10};
  if (buffer.Data == nullptr)
    return first_and_last;
  if (buffer.Data == first_and_last.Data)
    return XRayBuffer{nullptr, 0};
  assert(false && "Invalid buffer provided.");
}

static constexpr char Options[] = "additional_flags";

[[clang::xray_never_instrument]] XRayLogInitStatus
printing_init(size_t BufferSize, size_t MaxBuffers, void *Config,
              size_t ArgsSize) {
  // We require that the printing init is called through the
  // __xray_log_init_mode(...) implementation, and that the promised contract is
  // enforced.
  assert(BufferSize == 0);
  assert(MaxBuffers == 0);
  assert(Config != nullptr);
  assert(ArgsSize == 0 || ArgsSize == sizeof(Options));
  __xray_log_set_buffer_iterator(next_buffer);
  return XRayLogInitStatus::XRAY_LOG_INITIALIZED;
}

[[clang::xray_never_instrument]] XRayLogInitStatus printing_finalize() {
  return XRayLogInitStatus::XRAY_LOG_FINALIZED;
}

[[clang::xray_never_instrument]] XRayLogFlushStatus printing_flush_log() {
  __xray_log_remove_buffer_iterator();
  return XRayLogFlushStatus::XRAY_LOG_FLUSHED;
}

[[clang::xray_always_instrument]] void callme() { std::printf("called me!\n"); }

static auto buffer_counter = 0;

void process_buffer(const char *, XRayBuffer) { ++buffer_counter; }

int main(int argc, char **argv) {
  assert(__xray_log_register_mode("custom",
                                  {printing_init, printing_finalize,
                                   printing_handler, printing_flush_log}) ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  assert(__xray_log_select_mode("custom") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  assert(__xray_log_get_current_mode() != nullptr);
  std::string current_mode = __xray_log_get_current_mode();
  assert(current_mode == "custom");
  assert(__xray_patch() == XRayPatchingStatus::SUCCESS);
  assert(__xray_log_init_mode("custom", "flags_config_here=true") ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);

  // Also test that we can use the "binary" version of the
  // __xray_log_niit_mode(...) API.
  assert(__xray_log_init_mode_bin("custom", Options, sizeof(Options)) ==
         XRayLogInitStatus::XRAY_LOG_INITIALIZED);

  // CHECK: printing {{.*}}
  callme(); // CHECK: called me!
  // CHECK: printing {{.*}}
  assert(__xray_log_finalize() == XRayLogInitStatus::XRAY_LOG_FINALIZED);
  assert(__xray_log_process_buffers(process_buffer) ==
         XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  assert(buffer_counter == 1);
  assert(__xray_log_flushLog() == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  assert(__xray_log_select_mode("not-found") ==
         XRayLogRegisterStatus::XRAY_MODE_NOT_FOUND);
}
