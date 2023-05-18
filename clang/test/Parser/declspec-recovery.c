// RUN: %clang_cc1 -triple i386-pc-unknown -fsyntax-only -verify %s

__declspec(naked) void f(void) {} // expected-error{{'__declspec' attributes are not enabled; use '-fdeclspec' or '-fms-extensions' to enable support for __declspec attributes}}

struct S {
  __declspec(property(get=Getter, put=Setter)) int X; // expected-error{{'__declspec' attributes are not enabled; use '-fdeclspec' or '-fms-extensions' to enable support for __declspec attributes}}
  int Y;
};
