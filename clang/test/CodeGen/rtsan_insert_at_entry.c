// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsanitize=realtime -emit-llvm -o - %s | FileCheck %s

// The first instruction after the function is entred should be a call to
// enable the realtime sanitizer stack

int foo(int *a) [[clang::nonblocking]] { return *a; }
// CHECK-LABEL: define{{.*}}@foo
// CHECK-NEXT: entry:
// CHECK-NEXT: call{{.*}}__rtsan_realtime_enter
