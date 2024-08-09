// RUN: %clang     -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -flto -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -flto -o - | FileCheck %s

// Ensure the rtsan_realtime calls are never optimized away

int foo(int *a) [[clang::nonblocking]] { return *a; }
// CHECK: __rtsan_realtime_enter
// CHECK: __rtsan_realtime_exit
