// Test that HWASan and KHWASan runs with the new pass manager.
// We run them under different optimizations to ensure the IR is still
// being instrumented properly.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=hwaddress %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=hwaddress -O1 %s | FileCheck %s

// Don't instrument when collecting profiles, to avoid additional slowdown of slow `profile-instrument` binary.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=hwaddress -fprofile-instrument=llvm %s | FileCheck %s -check-prefixes=NOHWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=hwaddress -fprofile-instrument=llvm -O1 %s | FileCheck %s -check-prefixes=NOHWASAN

// Nothing special is done for clang PGO.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=hwaddress -fprofile-instrument=clang %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=hwaddress -fprofile-instrument=clang -O1 %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=kernel-hwaddress %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=kernel-hwaddress -O1 %s | FileCheck %s

// Don't instrument when collecting profiles, to avoid additional slowdown of slow `profile-instrument` binary.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=kernel-hwaddress -fprofile-instrument=llvm %s | FileCheck %s -check-prefixes=NOHWASAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=kernel-hwaddress -fprofile-instrument=llvm -O1 %s | FileCheck %s -check-prefixes=NOHWASAN

// Nothing special is done for clang PGO.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=kernel-hwaddress -fprofile-instrument=clang %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm -o - -fsanitize=kernel-hwaddress -fprofile-instrument=clang -O1 %s | FileCheck %s

int foo(int *a) { return *a; }

// All the cases above mark the function with sanitize_hwaddress.
// CHECK: sanitize_hwaddress
// CHECK: declare void @__hwasan_
// NOHWASAN-NOT: __hwasan
