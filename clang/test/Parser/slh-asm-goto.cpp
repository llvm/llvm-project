// RUN: %clang_cc1 -mspeculative-load-hardening -fsyntax-only -verify %s

void f() {
  __asm goto("movl %ecx, %edx"); // expected-warning {{speculative load hardening does not protect functions with asm goto}}
}
