// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S                       -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=address    -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=bounds     -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,BOUNDS
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=memory     -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,MSAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=thread     -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,TSAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=undefined  -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,UBSAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=kcfi       -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,KCFI

int x[10];
extern void (*f)(void);

// CHECK-LABEL: define dso_local void @foo(
void foo(int n) {
  // CHECK-DAG: call void @__sanitizer_cov_trace_pc
  // CHECK-DAG: call void @__sanitizer_cov_trace_const_cmp
  // ASAN-DAG: call void @__asan_report_store
  // MSAN-DAG: call void @__msan_warning
  // BOUNDS-DAG: call void @__ubsan_handle_out_of_bounds
  // TSAN-DAG: call void @__tsan_func_entry
  // UBSAN-DAG: call void @__ubsan_handle
  if (n)
    x[n] = 42;
  // KCFI-DAG: call void %[[#]]() [ "kcfi"(i32 {{.*}}) ]
  f();
}

static inline __attribute__((__always_inline__)) void always_inlined_fn(int n) {
  if (n)
    x[n] = 42;
}
// CHECK-LABEL: define dso_local void @test_always_inline(
void test_always_inline(int n) {
  // CHECK-DAG: call void @__sanitizer_cov_trace_pc
  // CHECK-DAG: call void @__sanitizer_cov_trace_const_cmp
  always_inlined_fn(n);
}

// CHECK-LABEL: define dso_local void @test_no_sanitize_coverage(
__attribute__((no_sanitize("coverage"))) void test_no_sanitize_coverage(int n) {
  // CHECK-NOT: call void @__sanitizer_cov_trace_pc
  // CHECK-NOT: call void @__sanitizer_cov_trace_const_cmp
  // ASAN-DAG: call void @__asan_report_store
  // MSAN-DAG: call void @__msan_warning
  // BOUNDS-DAG: call void @__ubsan_handle_out_of_bounds
  // TSAN-DAG: call void @__tsan_func_entry
  // UBSAN-DAG: call void @__ubsan_handle
  if (n)
    x[n] = 42;
  // KCFI-DAG: call void %[[#]]() [ "kcfi"(i32 {{.*}}) ]
  f();
}


// CHECK-LABEL: define dso_local void @test_no_sanitize_combined(
__attribute__((no_sanitize("address", "memory", "thread", "bounds", "undefined", "coverage")))
void test_no_sanitize_combined(int n) {
  // CHECK-NOT: call void @__sanitizer_cov_trace_pc
  // CHECK-NOT: call void @__sanitizer_cov_trace_const_cmp
  // ASAN-NOT: call void @__asan_report_store
  // MSAN-NOT: call void @__msan_warning
  // BOUNDS-NOT: call void @__ubsan_handle_out_of_bounds
  // BOUNDS-NOT: call void @llvm.trap()
  // TSAN-NOT: call void @__tsan_func_entry
  // UBSAN-NOT: call void @__ubsan_handle
  if (n)
    x[n] = 42;
}

// CHECK-LABEL: define dso_local void @test_no_sanitize_separate(
__attribute__((no_sanitize("address")))
__attribute__((no_sanitize("memory")))
__attribute__((no_sanitize("thread")))
__attribute__((no_sanitize("bounds")))
__attribute__((no_sanitize("undefined")))
__attribute__((no_sanitize("coverage")))
void test_no_sanitize_separate(int n) {
  // CHECK-NOT: call void @__sanitizer_cov_trace_pc
  // CHECK-NOT: call void @__sanitizer_cov_trace_const_cmp
  // ASAN-NOT: call void @__asan_report_store
  // MSAN-NOT: call void @__msan_warning
  // BOUNDS-NOT: call void @__ubsan_handle_out_of_bounds
  // BOUNDS-NOT: call void @llvm.trap()
  // TSAN-NOT: call void @__tsan_func_entry
  // UBSAN-NOT: call void @__ubsan_handle
  if (n)
    x[n] = 42;
}

// CHECK-LABEL: define dso_local void @test_no_sanitize_kcfi(
__attribute__((no_sanitize("kcfi", "coverage")))
void test_no_sanitize_kcfi(void) {
  // CHECK-NOT: call void @__sanitizer_cov_trace
  // KCFI-NOT: call void %[[#]]() [ "kcfi"(i32 {{.*}}) ]
  f();
}

// CHECK-LABEL: define dso_local void @test_no_sanitize_always_inline(
__attribute__((no_sanitize("coverage")))
void test_no_sanitize_always_inline(int n) {
  // CHECK-NOT: call void @__sanitizer_cov_trace_pc
  // CHECK-NOT: call void @__sanitizer_cov_trace_const_cmp
  always_inlined_fn(n);
}

// CHECK-LABEL: declare void
