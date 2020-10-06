// Test that HWASan and KHWASan runs with the new pass manager.
// We run them under different optimizations to ensure the IR is still
// being instrumented properly.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=hwaddress %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=hwaddress %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress %s | FileCheck %s

int foo(int *a) { return *a; }

// All the cases above mark the function with sanitize_hwaddress.
// CHECK: sanitize_hwaddress
