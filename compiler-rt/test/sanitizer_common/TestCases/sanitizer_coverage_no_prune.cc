// Tests -fsanitize-coverage=no-prune

// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: i386-darwin
// XFAIL: ubsan,tsan
// XFAIL: android && i386-target-arch && asan

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=trace-pc,bb,no-prune 2>&1 | grep "call void @__sanitizer_cov_trace_pc" | count 3
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=trace-pc,bb          2>&1 | grep "call void @__sanitizer_cov_trace_pc" | count 2
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=trace-pc,no-prune    2>&1 | grep "call void @__sanitizer_cov_trace_pc" | count 4
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=trace-pc             2>&1 | grep "call void @__sanitizer_cov_trace_pc" | count 3

void foo(int *a) {
  if (a)
    *a = 1;
}
