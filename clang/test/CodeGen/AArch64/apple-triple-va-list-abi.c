// RUN: %clang_cc1 -triple arm64-apple-macos -emit-llvm -o - %s | FileCheck %s --check-prefix=DARWINPCS
// RUN: %clang_cc1 -triple arm64-apple-macos -x c++ -std=c++23 -emit-llvm -o - %s | FileCheck %s --check-prefix=DARWINPCS
// RUN: %clang_cc1 -triple arm64-apple-macos -target-abi aapcs -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple arm64-apple-macos -target-abi darwinpcs -emit-llvm -o - %s | FileCheck %s --check-prefix=DARWINPCS

int read_variadic_int(int first, ...) {
  __builtin_va_list list;
  __builtin_va_start(list, first);
  int value = __builtin_va_arg(list, int);
  __builtin_va_end(list);
  return value;
}

// AAPCS-LABEL: define {{.*}}i32 @{{.*}}read_variadic_int
// AAPCS: %list = alloca %{{.*}}__va_list
// AAPCS: vaarg.maybe_reg:

// DARWINPCS-LABEL: define {{.*}}i32 @{{.*}}read_variadic_int
// DARWINPCS: %list = alloca ptr
// DARWINPCS: va_arg ptr %list, i32
