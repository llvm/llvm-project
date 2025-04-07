// RUN: %clang_cc1 %s -verify -fsyntax-only

struct Foo {
  virtual ~Foo() {} // expected-error {{attempt to use a deleted function}}
  static void operator delete(void* ptr) = delete; // expected-note {{explicitly marked deleted here}}
};


struct Bar {
  virtual ~Bar() {}
  static void operator delete[](void* ptr) = delete;
};

struct Baz {
  virtual ~Baz() {}
  static void operator delete[](void* ptr) = delete; // expected-note {{explicitly marked deleted here}}
};

void foobar() {
  Baz *B = new Baz[10]();
  delete [] B; // expected-error {{attempt to use a deleted function}}
}
