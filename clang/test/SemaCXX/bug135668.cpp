// RUN: %clang_cc1 -triple arm64-apple-macosx -Wall -fsyntax-only -verify %s -std=c++26 -fexceptions -fcxx-exceptions
// expected-no-diagnostics

// This test makes sure that we don't erroneously consider an accessible operator
// delete to be inaccessible, and then discard the entire new expression.

class TestClass {
public:
  TestClass();
  int field = 0;
  friend class Foo;
  static void * operator new(unsigned long size);
private:
  static void operator delete(void *p);
};

class Foo {
public:
  int test_method();
};

int Foo::test_method() {
  TestClass *obj = new TestClass() ;
  return obj->field;
}
