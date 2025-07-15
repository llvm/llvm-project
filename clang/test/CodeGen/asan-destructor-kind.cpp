// Frontend rejects invalid option
// RUN: not %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor=bad_arg -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CHECK-BAD-ARG
// CHECK-BAD-ARG: invalid value 'bad_arg' in '-fsanitize-address-destructor=bad_arg'

// Default is global dtor
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - -triple x86_64-apple-macosx10.15 %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-GLOBAL-DTOR

// Explicitly ask for global dtor
// RUN: %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor=global -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK-GLOBAL-DTOR

// CHECK-GLOBAL-DTOR: llvm.global_dtor{{.+}}asan.module_dtor
// CHECK-GLOBAL-DTOR: define internal void @asan.module_dtor

// Explicitly ask for no dtors
// RUN: %clang_cc1 -fsanitize=address \
// RUN:   -fsanitize-address-destructor=none -emit-llvm -o - \
// RUN:   -triple x86_64-apple-macosx10.15 %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK-NONE-DTOR

int global;

int main() {
  return global;
}

// CHECK-NONE-DTOR-NOT: llvm.global_dtor{{.+}}asan.module_dtor
// CHECK-NONE-DTOR-NOT: define internal void @asan.module_dtor
