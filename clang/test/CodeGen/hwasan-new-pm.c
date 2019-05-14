// Test that HWASan and KHWASan runs with the new pass manager.
// We run them under different optimizations and LTOs to ensure the IR is still
// being instrumented properly.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=hwaddress %s | FileCheck %s --check-prefixes=CHECK,HWASAN,HWASAN-NOOPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=hwaddress -flto %s | FileCheck %s --check-prefixes=CHECK,HWASAN,HWASAN-NOOPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=hwaddress -flto=thin %s | FileCheck %s --check-prefixes=CHECK,HWASAN,HWASAN-NOOPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=hwaddress %s | FileCheck %s --check-prefixes=CHECK,HWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=hwaddress -flto %s | FileCheck %s --check-prefixes=CHECK,HWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=hwaddress -flto=thin %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress %s | FileCheck %s --check-prefixes=CHECK,KHWASAN,KHWASAN-NOOPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress -flto %s | FileCheck %s --check-prefixes=CHECK,KHWASAN,KHWASAN-NOOPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress -flto=thin %s | FileCheck %s --check-prefixes=CHECK,KHWASAN,KHWASAN-NOOPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress %s | FileCheck %s --check-prefixes=CHECK,KHWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress -flto %s | FileCheck %s --check-prefixes=CHECK,KHWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=kernel-hwaddress -flto=thin %s | FileCheck %s

int foo(int *a) { return *a; }

// All the cases above mark the function with sanitize_hwaddress.
// CHECK-DAG: sanitize_hwaddress

// Both sanitizers produce %hwasan.shadow without both thinlto and optimizations.
// HWASAN-DAG: %hwasan.shadow
// KHWASAN-DAG: %hwasan.shadow

// Both sanitizers produce __hwasan_tls without both thinlto and optimizations.
// HWASAN-DAG: __hwasan_tls
// KHWASAN-DAG: __hwasan_tls

// For unoptimized cases, both sanitizers produce different load functions.
// HWASAN-NOOPT-DAG: __hwasan_loadN
// KHWASAN-NOOPT-DAG: __hwasan_loadN_noabort
