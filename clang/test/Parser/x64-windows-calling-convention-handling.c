// RUN: %clang_cc1 -triple x86_64-windows -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-mingw   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-cygwin  -fsyntax-only -verify %s

int __cdecl cdecl(int a, int b, int c, int d) { // expected-no-diagnostics
  return a + b + c + d;
}

float __stdcall stdcall(float a, float b, float c, float d) { // expected-no-diagnostics
  return a + b + c + d;
}
