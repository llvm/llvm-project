// RUN: %clang     -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s

// __rtsan_realtime_exit should be inserted at all function returns

int bar(int* x) [[clang::nonblocking]] {
  return *x;
}
// CHECK: __rtsan_realtime_exit
// CHECK-NEXT: ret
