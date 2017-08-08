// RUN: %clang_cc1 -fsyntax-only -Wunreachable-code -verify %s

static const bool False = false;

struct A {
  ~A();
  operator bool();
};
void Bar();

void Foo() {
  if (False && A()) {
    Bar(); // expected-no-diagnostics
  }
}
