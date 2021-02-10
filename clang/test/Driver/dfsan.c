// RUN: %clang     -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang -O2 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto -o - | FileCheck %s
// RUN: %clang -O2 -fexperimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto -o - | FileCheck %s

// RUN: %clang     -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto -o - | FileCheck %s
// RUN: %clang -O2 -fno-experimental-new-pass-manager -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto -o - | FileCheck %s

// Verify that -fsanitize=dataflow invokes DataFlowSanitizerPass instrumentation.

int foo(int *a) { return *a; }
// CHECK: __dfsan
