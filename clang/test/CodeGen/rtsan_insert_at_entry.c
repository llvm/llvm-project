// RUN: %clang     -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s

// The first instruction after the function is entred should be a call to
// enable the realtime sanitizer stack

int foo(int *a) [[clang::nonblocking]] { return *a; }
// CHECK: define{{.*}}foo
// CHECK-NEXT: __rtsan_realtime_enter
