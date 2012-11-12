// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// <rdar://problem/10228639>
class Foo {
  ~Foo();
  Foo(const Foo&);
public:
  Foo(int);
};

class Bar {
  int foo_count;
  Foo foos[0];
  Foo foos2[0][2];
  Foo foos3[2][0];

public:
  Bar(): foo_count(0) { }    
  ~Bar() { }
};

void testBar() {
  Bar b;
  Bar b2(b);
  b = b2;
}
