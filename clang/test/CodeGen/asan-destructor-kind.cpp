// Frontend rejects invalid option
// RUN: not %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=bad_arg -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK-BAD-ARG
// CHECK-BAD-ARG: unsupported argument 'bad_arg' to option 'fsanitize-address-destructor-kind='

// Default is global dtor
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-apple-macosx10.15 \
// RUN:   -fno-legacy-pass-manager %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-GLOBAL-DTOR
//
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-apple-macosx10.15 \
// RUN:   -flegacy-pass-manager %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-GLOBAL-DTOR

// Explictly ask for global dtor
// RUN: %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=global -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 -fno-legacy-pass-manager %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK-GLOBAL-DTOR
//
// RUN: %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=global -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 -flegacy-pass-manager %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK-GLOBAL-DTOR

// CHECK-GLOBAL-DTOR: llvm.global_dtor{{.+}}asan.module_dtor
// CHECK-GLOBAL-DTOR: define internal void @asan.module_dtor

// Explictly ask for no dtors
// RUN: %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=none -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 -fno-legacy-pass-manager %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK-NONE-DTOR
//
// RUN: %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor-kind=none -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 -flegacy-pass-manager %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK-NONE-DTOR

int global;

int main() {
  return global;
}

// CHECK-NONE-DTOR-NOT: llvm.global_dtor{{.+}}asan.module_dtor
// CHECK-NONE-DTOR-NOT: define internal void @asan.module_dtor
