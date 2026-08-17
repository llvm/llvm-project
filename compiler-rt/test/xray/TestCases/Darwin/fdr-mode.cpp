// Verify that XRay FDR (Flight Data Recorder) mode works on macOS.

// RUN: %clangxx_xray -fxray-instruction-threshold=1 %s -o %t
// RUN: rm -f %t.fdr-log-*
// RUN: env XRAY_OPTIONS="patch_premain=false xray_mode=xray-fdr verbosity=1 \
// RUN:     xray_logfile_base=%t.fdr-log-" \
// RUN:   XRAY_FDR_OPTIONS="func_duration_threshold_us=0" \
// RUN:   %run %t 2>&1 | FileCheck %s
// RUN: %llvm_xray convert --symbolize --output-format=yaml \
// RUN:   --instr_map=%t %t.fdr-log-* | FileCheck %s --check-prefix TRACE

// REQUIRES: target={{(arm64|x86_64)-apple-.*}}

// CHECK: init_status=2
// CHECK: finalize_status=4

// TRACE: records:
// TRACE: kind: function-enter

#include "xray/xray_interface.h"
#include "xray/xray_log_interface.h"
#include <cstdio>

[[clang::xray_always_instrument]] int fdr_target(int x) { return x + 1; }

int main() {
  __xray_log_select_mode("xray-fdr");
  auto init_status =
      __xray_log_init_mode("xray-fdr", "buffer_size=16384:buffer_max=10");
  printf("init_status=%d\n", (int)init_status);
  __xray_patch();
  for (int i = 0; i < 10; ++i)
    fdr_target(i);
  auto finalize_status = __xray_log_finalize();
  printf("finalize_status=%d\n", (int)finalize_status);
  __xray_log_flushLog();
  return 0;
}
