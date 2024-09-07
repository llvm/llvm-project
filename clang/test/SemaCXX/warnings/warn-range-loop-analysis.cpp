// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wall -Wrange-loop-bind-reference -Wno-unused -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wloop-analysis -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wrange-loop-analysis -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -Wloop-analysis -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

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
  // Small trivially copyable types do not show a warning when copied in a
  // range-based for loop. This size ensures the object is not considered
  // small.
  char s[128];
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
  // expected-warning@-1 {{loop variable 'x' binds to a temporary value produced by a range of type 'Container<int>'}}
  // expected-note@-2 {{use non-reference type 'int'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""

  for (const double &x : int_container) {}
  // expected-warning@-1 {{loop variable 'x' of type 'const double &' binds to a temporary constructed from type 'int &'}}
  // expected-note@-2 {{use non-reference type 'double' to make construction explicit or type 'const int &' to prevent copying}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:22}:""

  for (const Bar x : bar_container) {}
  // expected-warning@-1 {{loop variable 'x' creates a copy from type 'const Bar'}}
  // expected-note@-2 {{use reference type 'const Bar &' to prevent copying}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:18}:"&"
}

void test1() {
  Container<int> A;

  for (const int &&x : A) {}
  // No warning, rvalue-reference to the temporary
  for (const int &x : A) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'int'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const int x : A) {}
  // No warning, non-reference type indicates copy is made
  for (int&& x : A) {}
  // No warning, rvalue-reference to the temporary
  //for (int &x : A) {}
  // Binding error
  for (int x : A) {}
  // No warning, non-reference type indicates copy is made

  for (const double &&x : A) {}
  // No warning, rvalue-reference to the temporary
  for (const double &x : A) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'double'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:22}:""
  for (const double x : A) {}
  // No warning, non-reference type indicates copy is made
  for (double &&x : A) {}
  // No warning, rvalue-reference to the temporary
  //for (double &x : A) {}
  // Binding error
  for (double x : A) {}
  // No warning, non-reference type indicates copy is made

  for (const Bar &&x : A) {}
  // No warning, rvalue-reference to the temporary
  for (const Bar &x : A) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : A) {}
  // No warning, non-reference type indicates copy is made
  for (Bar &&x : A) {}
  // No warning, rvalue-reference to the temporary
  //for (Bar &x : A) {}
  // Binding error
  for (Bar x : A) {}
  // No warning, non-reference type indicates copy is made
}

void test2() {
  Container<int&> B;

  //for (const int &&x : B) {}
  // Binding error
  for (const int &x : B) {}
  // No warning, this reference is not a temporary
  for (const int x : B) {}
  // No warning on POD copy
  //for (int &x : B) {}
  // Binding error
  for (int &x : B) {}
  // No warning
  for (int x : B) {}
  // No warning

  for (const double &&x : B) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'double'{{.*}}'const int &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:23}:""
  for (const double &x : B) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'double'{{.*}}'const int &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:21-[[@LINE-3]]:22}:""
  for (const double x : B) {}
  for (double &&x : B) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'double'{{.*}}'const int &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:17}:""
  //for (double &x : B) {}
  // Binding error
  for (double x : B) {}
  // No warning

  for (const Bar &&x : B) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const Bar &x : B) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : B) {}
  for (Bar &&x : B) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (Bar &x : B) {}
  // Binding error
  for (Bar x : B) {}
  // No warning
}

void test3() {
  Container<Bar> C;

  for (const Bar &&x : C) {}
  // No warning, rvalue-reference to the temporary
  for (const Bar &x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : C) {}
  // No warning, non-reference type indicates copy is made
  for (Bar &&x : C) {}
  // No warning, rvalue-reference to the temporary
  //for (Bar &x : C) {}
  // Binding error
  for (Bar x : C) {}
  // No warning, non-reference type indicates copy is made

  for (const int &&x : C) {}
  // No warning, rvalue-reference to the temporary
  for (const int &x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'int'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const int x : C) {}
  // No warning, copy made
  for (int &&x : C) {}
  // No warning, rvalue-reference to the temporary
  //for (int &x : C) {}
  // Binding error
  for (int x : C) {}
  // No warning, copy made
}

void test4() {
  Container<Bar&> D;

  //for (const Bar &&x : D) {}
  // Binding error
  for (const Bar &x : D) {}
  // No warning, this reference is not a temporary
  for (const Bar x : D) {}
  // expected-warning@-1 {{creates a copy}}
  // expected-note@-2 {{'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:18}:"&"
  //for (Bar &&x : D) {}
  // Binding error
  for (Bar &x : D) {}
  // No warning
  for (Bar x : D) {}
  // No warning

  for (const int &&x : D) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const int &x : D) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const int x : D) {}
  // No warning
  for (int &&x : D) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (int &x : D) {}
  // Binding error
  for (int x : D) {}
  // No warning
}

void test5() {
  Container<Foo> E;

  for (const Bar &&x : E) {}
  // No warning, rvalue-reference to the temporary
  for (const Bar &x : E) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : E) {}
  // No warning, non-reference type indicates copy is made
  for (Bar &&x : E) {}
  // No warning, rvalue-reference to the temporary
  //for (Bar &x : E) {}
  // Binding error
  for (Bar x : E) {}
  // No warning, non-reference type indicates copy is made
}

void test6() {
  Container<Foo&> F;

  for (const Bar &&x : F) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const Bar &x : F) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : F) {}
  // No warning.
  for (Bar &&x : F) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (Bar &x : F) {}
  // Binding error
  for (Bar x : F) {}
  // No warning
}

void test7() {
  double G[2];

  //for (const double &&x : G) {}
  // Binding error
  for (const double &x : G) {}
  // No warning
  for (const double x : G) {}
  // No warning on POD copy
  //for (double &&x : G) {}
  // Binding error
  for (double &x : G) {}
  // No warning
  for (double x : G) {}
  // No warning

  for (const int &&x : G) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const double &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const int &x : G) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const double &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const int x : G) {}
  // No warning
  for (int &&x : G) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const double &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (int &x : G) {}
  // Binding error
  for (int x : G) {}
  // No warning

  for (const Bar &&x : G) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const double &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const Bar &x : G) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const double &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : G) {}
  // No warning
  for (Bar &&x : G) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const double &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (Bar &x : G) {}
  // Binding error
  for (Bar x : G) {}
  // No warning
}

void test8() {
  Foo H[2];

  //for (const Foo &&x : H) {}
  // Binding error
  for (const Foo &x : H) {}
  // No warning
  for (const Foo x : H) {}
  // No warning on POD copy
  //for (Foo &&x : H) {}
  // Binding error
  for (Foo &x : H) {}
  // No warning
  for (Foo x : H) {}
  // No warning

  for (const Bar &&x : H) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const Bar &x : H) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const Bar x : H) {}
  // No warning
  for (Bar &&x: H) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'Bar'{{.*}}'const Foo &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (Bar &x: H) {}
  // Binding error
  for (Bar x: H) {}
  // No warning
}

void test9() {
  Bar I[2] = {1,2};

  //for (const Bar &&x : I) {}
  // Binding error
  for (const Bar &x : I) {}
  // No warning
  for (const Bar x : I) {}
  // expected-warning@-1 {{creates a copy}}
  // expected-note@-2 {{'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:18}:"&"
  //for (Bar &&x : I) {}
  // Binding error
  for (Bar &x : I) {}
  // No warning
  for (Bar x : I) {}
  // No warning

  for (const int &&x : I) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""
  for (const int &x : I) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""
  for (const int x : I) {}
  // No warning
  for (int &&x : I) {}
  // expected-warning@-1 {{binds to a temporary constructed from}}
  // expected-note-re@-2 {{'int'{{.*}}'const Bar &'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:12-[[@LINE-3]]:14}:""
  //for (int &x : I) {}
  // Binding error
  for (int x : I) {}
  // No warning
}

void test10() {
  Container<Bar> C;

  for (const Bar &x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""

  for (const Bar& x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:""

  for (const Bar & x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:20}:""

  for (const Bar&x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:17-[[@LINE-3]]:18}:" "
}

template <class T>
void test_template_function() {
  // In a template instantiation the diagnostics should not be emitted for
  // loops with dependent types.
  Container<Bar> C;
  for (const Bar &x : C) {}
  // expected-warning@-1 {{binds to a temporary value produced by a range}}
  // expected-note@-2 {{'Bar'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:18-[[@LINE-3]]:19}:""

  Container<T> Dependent;
  for (const T &x : Dependent) {}
}
template void test_template_function<Bar>();

template <class T>
struct test_template_struct {
  static void static_member() {
    Container<Bar> C;
    for (const Bar &x : C) {}
    // expected-warning@-1 {{binds to a temporary value produced by a range}}
    // expected-note@-2 {{'Bar'}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:21}:""

    Container<T> Dependent;
    for (const T &x : Dependent) {}
  }

  void member() {
    Container<Bar> C;
    for (const Bar &x : C) {}
    // expected-warning@-1 {{binds to a temporary value produced by a range}}
    // expected-note@-2 {{'Bar'}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:21}:""

    Container<T> Dependent;
    for (const T &x : Dependent) {}
  }
};
template struct test_template_struct<Bar>;

struct test_struct_with_templated_member {
  void member() {
    Container<Bar> C;
    for (const Bar &x : C) {}
    // expected-warning@-1 {{binds to a temporary value produced by a range}}
    // expected-note@-2 {{'Bar'}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:21}:""
  }

  template <class T>
  void template_member() {
    Container<Bar> C;
    for (const Bar &x : C) {}
    // expected-warning@-1 {{binds to a temporary value produced by a range}}
    // expected-note@-2 {{'Bar'}}
    // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:21}:""

    Container<T> Dependent;
    for (const T &x : Dependent) {}
  }
};
template void test_struct_with_templated_member::template_member<Bar>();

#define TEST_MACRO            \
  void test_macro() {         \
    Container<Bar> C;         \
    for (const Bar &x : C) {} \
  }

TEST_MACRO
