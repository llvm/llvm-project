// REQUIRES: x86-registered-target

// RUN: %clang     -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN

// RUN: %clang     -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KMSAN
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KMSAN
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KMSAN
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=kernel-memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-KMSAN

// RUN: %clang -target mips64-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -target mips64el-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -target powerpc64-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -target powerpc64le-unknown-linux-gnu -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN

// Verify that -fsanitize=memory and -fsanitize=kernel-memory invoke MSan/KMSAN instrumentation.

// Also check that this works with the new pass manager with and without
// optimization
// RUN: %clang     -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -O1 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -O2 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN
// RUN: %clang -O3 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=memory %s -S -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-MSAN

int foo(int *a) { return *a; }
// CHECK-MSAN: __msan_init
// CHECK-KMSAN: __msan_get_context_state
