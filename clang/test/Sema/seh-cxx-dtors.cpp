// RUN: %clang_cc1 -triple=x86_64-pc-win32 -fsyntax-only -fms-extensions -verify %s

struct Foo {
  ~Foo() {}
};

// These need '-fms-extensions'
void f1() {
  Foo foo {}; // expected-error {{destructor}}
  __try {} __except (1) {}
}

void bar(Foo foo);

void f2() {
  bar(Foo {}); // expected-error {{destructor}}
  __try {} __except (1) {}
}
