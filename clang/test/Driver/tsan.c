// REQUIRES: x86-registered-target

// RUN: %clang     -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize=thread  %s -S -emit-llvm -o - | FileCheck %s
// Verify that -fsanitize=thread invokes tsan instrumentation.

// Also check that this works with the new pass manager with and without
// optimization
// RUN: %clang     -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=thread %s -S -emit-llvm -o - | FileCheck %s

int foo(int *a) { return *a; }
// CHECK: __tsan_init
