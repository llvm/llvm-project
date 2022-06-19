// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions -fsyntax-only -verify %s

[[deprecated]] void f() {} // expected-note 2 {{marked deprecated here}}

#define From__pragma() \
  __pragma(warning(push)) \
  __pragma(warning(disable:4996)) \
  f(); \
  __pragma(warning(pop))

void g() {
  f(); // expected-warning {{deprecated}}

#pragma warning(push)
#pragma warning(disable: 4996)
  f(); // no diag

#pragma warning(disable: 49960000)
#pragma warning(pop)

  f(); // expected-warning {{deprecated}}

  From__pragma(); // no diag
}
