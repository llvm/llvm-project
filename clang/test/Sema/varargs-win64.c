// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-pc-win32

void __attribute__((sysv_abi)) foo(int a, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, a); // expected-error {{'va_start' used in System V ABI function}}
}
