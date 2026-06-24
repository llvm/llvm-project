// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++

// expected-no-diagnostics

void test(void)
{
  asm("");
loop:
  asm goto(""::::loop);
}
