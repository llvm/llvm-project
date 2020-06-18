// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S                       -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=address    -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=bounds     -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,BOUNDS
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=memory     -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,MSAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=thread     -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,TSAN
// RUN: %clang %s -target x86_64-unknown-linux-gnu -emit-llvm -S -fsanitize=undefined  -fsanitize-coverage=trace-pc,trace-cmp -o - | FileCheck %s --check-prefixes=CHECK,UBSAN
//
// Host armv7 is currently unsupported: https://bugs.llvm.org/show_bug.cgi?id=46117
// XFAIL: armv7, thumbv7

int x[10];

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
}
// CHECK-LABEL: declare void
