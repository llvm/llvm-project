// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsanitize=realtime -emit-llvm -o - %s | FileCheck %s

// __rtsan_realtime_exit should be inserted at all function returns

int bar(int* x) [[clang::nonblocking]] {
  return *x;
}
// CHECK-LABEL: call{{.*}}__rtsan_realtime_exit
// CHECK-NEXT: ret
