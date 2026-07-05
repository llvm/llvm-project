// RUN: %clang_cc1 -Wno-slh-asm-goto -mspeculative-load-hardening -fsyntax-only -verify %s

void f() {
  __asm goto("movl %ecx, %edx"); // expected-no-diagnostics
}
