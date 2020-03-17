// REQUIRES: aarch64-registered-target

// Old pass manager.
// RUN: %clang     -fno-experimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-NO-SAFETY
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SAFETY
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SAFETY
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SAFETY

// New pass manager.
// RUN: %clang     -fexperimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-NO-SAFETY
// RUN: %clang -O1 -fexperimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SAFETY
// RUN: %clang -O2 -fexperimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SAFETY
// RUN: %clang -O3 -fexperimental-new-pass-manager -target aarch64-unknown-linux -march=armv8+memtag -fsanitize=memtag %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-SAFETY

int z;
__attribute__((noinline)) void use(int *p) { *p = z; }
int foo() { int x; use(&x); return x; }

// CHECK-NO-SAFETY: define dso_local i32 @foo()
// CHECK-NO-SAFETY: %{{.*}} = alloca i32, align 4{{$}}

// CHECK-SAFETY: define dso_local i32 @foo()
// CHECK-SAFETY: %{{.*}} = alloca i32, align 4, !stack-safe
