// RUN: %clang_cc1 -fsyntax-only -verify %s -triple aarch64-linux-gnu

void f1(int a, ...) {
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a); // expected-error {{'__builtin_ms_va_start' used in System V ABI function}}
}

void __attribute__((ms_abi)) f2(int a, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, a); // expected-error {{'va_start' used in Win64 ABI function}}
}
