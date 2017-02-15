// RUN: %clang_cc1 -triple i386-pc-unknown -fsyntax-only -fms-extensions -verify %s
// RUN: %clang_cc1 -triple i386-pc-unknown -fsyntax-only -fdeclspec -verify %s
// expected-no-diagnostics

__declspec(naked) void f(void) {}

struct S {
  __declspec(property(get=Getter, put=Setter)) int X;
  int Y;
};
