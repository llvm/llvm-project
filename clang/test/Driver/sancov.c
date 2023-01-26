// RUN: %clang     -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O1 -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang -O3 -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -flto=thin -o - | FileCheck %s
// RUN: %clang     -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -flto -o - | FileCheck %s
// RUN: %clang -O2 -target x86_64-unknown-linux -fsanitize-coverage=trace-pc-guard %s -S -emit-llvm -flto -o - | FileCheck %s

// Verify that -fsanitize-coverage invokes SanitizerCoveragePass instrumentation.

int foo(int *a) { return *a; }
// CHECK: _sancov_
