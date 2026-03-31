// COM: Test the AMD_COMGR_DRIVER_OPTIONS_APPEND environment variable
// COM: This test verifies that options set via the environment variable
// COM: are appended to clang driver invocations.

// RUN: env AMD_COMGR_EMIT_VERBOSE_LOGS=1 \
// RUN:     AMD_COMGR_REDIRECT_LOGS=stdout \
// RUN:     AMD_COMGR_DRIVER_OPTIONS_APPEND="-DTEST_MACRO=123 -Wno-extra" \
// RUN:     compile-hip-minimal %s %t.bin | \
// RUN:     %FileCheck %s

// CHECK: Compilation Args:
// CHECK-SAME: "-DTEST_MACRO=123"
// CHECK-SAME: "-Wno-extra"

__attribute__((global))
void test(int *out) {
  out[0] = 42;
}
