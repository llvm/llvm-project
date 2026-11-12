// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fsanitize=realtime -emit-llvm -o - %s | FileCheck %s

int foo(int *a) [[clang::nonblocking]] { return *a; }

// The first instruction after the function is entred should be a call to
// enable the realtime sanitizer stack.
// CHECK-LABEL: define{{.*}}@foo
// CHECK-NEXT: entry:
// CHECK-NEXT: call{{.*}}__rtsan_realtime_enter

// __rtsan_realtime_exit should be inserted at all function returns.
// CHECK-LABEL: call{{.*}}__rtsan_realtime_exit
// CHECK-NEXT: ret
