// RUN: %clang     -target x86_64-unknown-linux -fsanitize=efficiency-cache-frag %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize=efficiency-cache-frag %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=efficiency-cache-frag %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize=efficiency-cache-frag %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize=efficiency-working-set %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize=efficiency-working-set %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=efficiency-working-set %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize=efficiency-working-set %s -S -emit-llvm -o - | FileCheck %s
// Verify that -fsanitize=efficiency-* invokes esan instrumentation.

int foo(int *a) { return *a; }
// CHECK: __esan_init
