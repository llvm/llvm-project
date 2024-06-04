// REQUIRES: x86-registered-target

// RUN: %clang --target=mips64-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang --target=mips64el-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang --target=powerpc64-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang --target=powerpc64le-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK0

// Verify that -fsanitize=memory and -fsanitize=kernel-memory invoke MSan/KMSAN instrumentation.

// RUN: %clang     --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang -O1 --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang -O2 --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang -O2 --target=x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -flto -o - | FileCheck %s

// RUN: %clang     --target=x86_64-unknown-linux -fsanitize=kernel-memory  %s -S -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang -O1 --target=x86_64-unknown-linux -fsanitize=kernel-memory  %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 --target=x86_64-unknown-linux -fsanitize=kernel-memory  %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 --target=x86_64-unknown-linux -fsanitize=kernel-memory  %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     --target=x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -flto=thin -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang -O2 --target=x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     --target=x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -flto -o - | FileCheck %s --check-prefixes=CHECK0
// RUN: %clang -O2 --target=x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -flto -o - | FileCheck %s

int foo(int *a) { return *a; }
// CHECK0: = alloca
// CHECK0: @__msan_

// CHECK-NOT: = alloca
// CHECK: @__msan_
