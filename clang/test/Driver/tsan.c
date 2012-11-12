// RUN: %clang     -target i386-unknown-unknown -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target i386-unknown-unknown -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target i386-unknown-unknown -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target i386-unknown-unknown -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -target i386-unknown-unknown -fsanitize=thread  %s -S -emit-llvm -o - | FileCheck %s
// Verify that -fsanitize=thread invokes tsan instrumentation.

int foo(int *a) { return *a; }
// CHECK: __tsan_init
