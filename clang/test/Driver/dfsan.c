// RUN: %clang     -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize=dataflow %s -S -emit-llvm -flto -o - | FileCheck %s

// Verify that -fsanitize=dataflow invokes DataFlowSanitizerPass instrumentation.

int foo(int *a) { return *a; }
// CHECK: __dfsan
