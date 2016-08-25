// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wloop-analysis -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wrange-loop-analysis -verify %s

template <typename return_type>
struct Iterator {
  return_type operator*();
  Iterator operator++();
  bool operator!=(const Iterator);
};

template <typename T>
struct Container {
  typedef Iterator<T> I;

  I begin();
  I end();
};

struct Foo {};
struct Bar {
  Bar(Foo);
  Bar(int);
  operator int();
};

// Testing notes:
// test0 checks that the full text of the warnings and notes is correct.  The
//   rest of the tests checks a smaller portion of the text.
// test1-6 are set in pairs, the odd numbers are the non-reference returning
//   versions of the even numbers.
// test7-9 use an array instead of a range object
// tests use all four versions of the loop variable, const &T, const T, T&, and
//   T.  Versions producing errors and are commented out.
//
// Conversion chart:
//   double <=> int
//   int    <=> Bar
//   double  => Bar
//   Foo     => Bar
//
// Conversions during tests:
// test1-2
//   int => int
//   int => double
//   int => Bar
// test3-4
//   Bar => Bar
//   Bar => int
// test5-6
//   Foo => Bar
// test7
//   double => double
//   double => int
//   double => Bar
// test8
//   Foo => Foo
//   Foo => Bar
// test9
//   Bar => Bar
//   Bar => int

void test0() {
  Container<int> int_non_ref_container;
  Container<int&> int_container;
  Container<Bar&> bar_container;

  for (const int &x : int_non_ref_container) {}
  // expected-warning@-1 {{loop variable 'x' is always a copy because the range of type 'Container<int>' does not return a reference}}
  // expected-note@-2 {{use non-reference type 'int'}}

  for (const double &x : int_container) {}
  // expected-warning@-1 {{loop variable 'x' has type 'const double &' but is initialized with type 'int' resulting in a copy}}
  // expected-note@-2 {{use non-reference type 'double' to keep the copy or type 'const int &' to prevent copying}}

  for (const Bar x : bar_container) {}
  // expected-warning@-1 {{loop variable 'x' of type 'const Bar' creates a copy from type 'const Bar'}}
  // expected-note@-2 {{use reference type 'const Bar &' to prevent copying}}
}

void test1() {
  Container<int> A;

  for (const int &x : A) {}
  // expected-warning@-1 {{always a copy}}
  // expected-note@-2 {{'int'}}
  for (const int x : A) {}
  // No warning, non-reference type indicates copy is made
  //for (int &x : A) {}
  // Binding error
  for (int x : A) {}
  // No warning, non-reference type indicates copy is made

  for (const double &x : A) {}
  // expected-warning@-1 {{always a copy}}
  // expected-note@-2 {{'double'}}
  for (const double x : A) {}
  // No warning, non-reference type indicates copy is made
  //for (double &x : A) {}
  // Binding error
  for (double x : A) {}
  // No warning, non-reference type indicates copy is made

  for (const Bar &x : A) {}
  // expected-warning@-1 {{always a copy}}
  // expected-note@-2 {{'Bar'}}
  for (const Bar x : A) {}
  // No warning, non-reference type indicates copy is made
  //for (Bar &x : A) {}
  // Binding error
  for (Bar x : A) {}
  // No warning, non-reference type indicates copy is made
}

void test2() {
  Container<int&> B;

  for (const int &x : B) {}
  // No warning, this reference is not a temporary
  for (const int x : B) {}
  // No warning on POD copy
  for (int &x : B) {}
  // No warning
  for (int x : B) {}
  // No warning

  for (const double &x : B) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'double'{{.*}}'const int &'}}
  for (const double x : B) {}
  //for (double &x : B) {}
  // Binding error
  for (double x : B) {}
  // No warning

  for (const Bar &x : B) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note@-2 {{'Bar'}}
  for (const Bar x : B) {}
  //for (Bar &x : B) {}
  // Binding error
  for (Bar x : B) {}
  // No warning
}

void test3() {
  Container<Bar> C;

  for (const Bar &x : C) {}
  // expected-warning@-1 {{always a copy}}
  // expected-note@-2 {{'Bar'}}
  for (const Bar x : C) {}
  // No warning, non-reference type indicates copy is made
  //for (Bar &x : C) {}
  // Binding error
  for (Bar x : C) {}
  // No warning, non-reference type indicates copy is made

  for (const int &x : C) {}
  // expected-warning@-1 {{always a copy}}
  // expected-note@-2 {{'int'}}
  for (const int x : C) {}
  // No warning, copy made
  //for (int &x : C) {}
  // Binding error
  for (int x : C) {}
  // No warning, copy made
}

void test4() {
  Container<Bar&> D;

  for (const Bar &x : D) {}
  // No warning, this reference is not a temporary
  for (const Bar x : D) {}
  // expected-warning@-1 {{creates a copy}}
  // expected-note@-2 {{'const Bar &'}}
  for (Bar &x : D) {}
  // No warning
  for (Bar x : D) {}
  // No warning

  for (const int &x : D) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  for (const int x : D) {}
  // No warning
  //for (int &x : D) {}
  // Binding error
  for (int x : D) {}
  // No warning
}

void test5() {
  Container<Foo> E;

  for (const Bar &x : E) {}
  // expected-warning@-1 {{always a copy}}
  // expected-note@-2 {{'Bar'}}
  for (const Bar x : E) {}
  // No warning, non-reference type indicates copy is made
  //for (Bar &x : E) {}
  // Binding error
  for (Bar x : E) {}
  // No warning, non-reference type indicates copy is made
}

void test6() {
  Container<Foo&> F;

  for (const Bar &x : F) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  for (const Bar x : F) {}
  // No warning.
  //for (Bar &x : F) {}
  // Binding error
  for (Bar x : F) {}
  // No warning
}

void test7() {
  double G[2];

  for (const double &x : G) {}
  // No warning
  for (const double x : G) {}
  // No warning on POD copy
  for (double &x : G) {}
  // No warning
  for (double x : G) {}
  // No warning

  for (const int &x : G) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'int'{{.*}}'const double &'}}
  for (const int x : G) {}
  // No warning
  //for (int &x : G) {}
  // Binding error
  for (int x : G) {}
  // No warning

  for (const Bar &x : G) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const double &'}}
  for (const Bar x : G) {}
  // No warning
  //for (int &Bar : G) {}
  // Binding error
  for (int Bar : G) {}
  // No warning
}

void test8() {
  Foo H[2];

  for (const Foo &x : H) {}
  // No warning
  for (const Foo x : H) {}
  // No warning on POD copy
  for (Foo &x : H) {}
  // No warning
  for (Foo x : H) {}
  // No warning

  for (const Bar &x : H) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  for (const Bar x : H) {}
  // No warning
  //for (Bar &x: H) {}
  // Binding error
  for (Bar x: H) {}
  // No warning
}

void test9() {
  Bar I[2] = {1,2};

  for (const Bar &x : I) {}
  // No warning
  for (const Bar x : I) {}
  // expected-warning@-1 {{creates a copy}}
  // expected-note@-2 {{'const Bar &'}}
  for (Bar &x : I) {}
  // No warning
  for (Bar x : I) {}
  // No warning

  for (const int &x : I) {}
  // expected-warning@-1 {{resulting in a copy}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  for (const int x : I) {}
  // No warning
  //for (int &x : I) {}
  // Binding error
  for (int x : I) {}
  // No warning
}
