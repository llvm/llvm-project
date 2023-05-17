// This will sometimes segfault on the AArch64 and Arm bots
// UNSUPPORTED: target={{(aarch64|arm).*}}
// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm xray-log.fdr-reinit* || true
// RUN: XRAY_OPTIONS="verbosity=1" %run %t
// RUN: rm xray-log.fdr-reinit* || true
#include "xray/xray_log_interface.h"
#include <atomic>
#include <cassert>
#include <cstddef>
#include <thread>

volatile uint64_t var = 0;

std::atomic_flag keep_going = ATOMIC_FLAG_INIT;

[[clang::xray_always_instrument]] void __attribute__((noinline)) func() {
  ++var;
}

int main(int argc, char *argv[]) {
  // Start a thread that will just keep calling the function, to spam calls to
  // the function call handler.
  keep_going.test_and_set(std::memory_order_acquire);
  std::thread t([] {
    while (keep_going.test_and_set(std::memory_order_acquire))
      func();
  });

  static constexpr char kConfig[] =
      "buffer_size=1024:buffer_max=10:no_file_flush=true";

  // Then we initialize the FDR mode implementation.
  assert(__xray_log_select_mode("xray-fdr") ==
         XRayLogRegisterStatus::XRAY_REGISTRATION_OK);
  auto init_status = __xray_log_init_mode("xray-fdr", kConfig);
  assert(init_status == XRayLogInitStatus::XRAY_LOG_INITIALIZED);

  // Now we patch the instrumentation points.
  __xray_patch();

  // Spin for a bit, calling func() enough times.
  for (auto i = 0; i < 1 << 20; ++i)
    func();

  // Then immediately finalize the implementation.
  auto finalize_status = __xray_log_finalize();
  assert(finalize_status == XRayLogInitStatus::XRAY_LOG_FINALIZED);

  // Once we're here, we should then flush.
  auto flush_status = __xray_log_flushLog();
  assert(flush_status == XRayLogFlushStatus::XRAY_LOG_FLUSHED);

  for (auto trial = 0; trial < 3; trial++) {
    // Without doing anything else, we should re-initialize.
    init_status = __xray_log_init_mode("xray-fdr", kConfig);
    assert(init_status == XRayLogInitStatus::XRAY_LOG_INITIALIZED);

    // Then we spin for a bit again calling func() enough times.
    for (auto i = 0; i < 1 << 20; ++i)
      func();

    // Then immediately finalize the implementation.
    finalize_status = __xray_log_finalize();
    assert(finalize_status == XRayLogInitStatus::XRAY_LOG_FINALIZED);

    // Once we're here, we should then flush.
    flush_status = __xray_log_flushLog();
    assert(flush_status == XRayLogFlushStatus::XRAY_LOG_FLUSHED);
  }

  // Finally, we should signal the sibling thread to stop.
  keep_going.clear(std::memory_order_release);

  // Then join.
  t.join();
}
