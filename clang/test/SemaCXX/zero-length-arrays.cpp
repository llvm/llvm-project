// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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
#if __cplusplus >= 201103L
// expected-note@-2 {{copy constructor of 'Bar' is implicitly deleted because field 'foos' has an inaccessible copy constructor}}
#endif
  Foo foos2[0][2];
  Foo foos3[2][0];

public:
  Bar(): foo_count(0) { }    
  ~Bar() { }
};

void testBar() {
  Bar b;
  Bar b2(b);
#if __cplusplus >= 201103L
// expected-error@-2 {{call to implicitly-deleted copy constructor of 'Bar}}
#else
// expected-no-diagnostics
#endif
  b = b2;
}
