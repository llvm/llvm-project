// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++1z -fsyntax-only -verify %s

struct pr12960 {
  int begin;
  void foo(int x) {
    for (int& it : x) { // expected-error {{invalid range expression of type 'int'; no viable 'begin' function available}}
    }
  }
};

struct null_t {
  operator int*();
};

namespace X {
  template<typename T>
    auto begin(T &&t) -> decltype(t.begin()) { return t.begin(); } // expected-note 2{{ignored: substitution failure}}
  template<typename T>
    auto end(T &&t) -> decltype(t.end()) { return t.end(); } // expected-note {{candidate template ignored: substitution failure [with T = }}

  template<typename T>
    auto begin(T &&t) -> decltype(t.alt_begin()) { return t.alt_begin(); } // expected-note {{selected 'begin' template [with T = }} \
                                                                              expected-note 2{{candidate template ignored: substitution failure [with T = }}
  template<typename T>
    auto end(T &&t) -> decltype(t.alt_end()) { return t.alt_end(); } // expected-note {{candidate template ignored: substitution failure [with T = }}

  namespace inner {
    // These should never be considered.
    int begin(int);
    int end(int);
  }

  using namespace inner;

  struct A { // expected-note 2 {{candidate constructor}}
    A();
    int *begin(); // expected-note 3{{selected 'begin' function with iterator type 'int *'}} expected-note {{'begin' declared here}}
    int *end();
  };

  struct B {
    B();
    int *alt_begin();
    int *alt_end();
  };

  struct NoBeginADL {
    null_t alt_end();
  };
  struct NoEndADL {
    null_t alt_begin();
  };

  struct C {
    C();
    struct It {
      int val;
      operator int &() { return val; }
    };
    It begin();
    It end();
  };

  constexpr int operator*(const C::It &) { return 0; }
}

using X::A;

void f();
void f(int);

void g() {
  for (int a : A())
    A __begin;
  for (char *a : A()) { // expected-error {{cannot initialize a variable of type 'char *' with an lvalue of type 'int'}}
  }
  for (char *a : X::B()) { // expected-error {{cannot initialize a variable of type 'char *' with an lvalue of type 'int'}}
  }
  // FIXME: Terrible diagnostic here. auto deduction should fail, but does not!
  for (double a : f) { // expected-error {{cannot use type '<overloaded function type>' as a range}}
  }
  for (auto a : A()) {
  }
  for (auto a : X::B()) {
  }
  for (auto *a : A()) { // expected-error {{variable 'a' with type 'auto *' has incompatible initializer of type 'int'}}
  }
  // : is not a typo for :: here.
  for (A NS:A()) { // expected-error {{no viable conversion from 'int' to 'X::A'}}
  }
  for (auto not_in_scope : not_in_scope) { // expected-error {{use of undeclared identifier 'not_in_scope'}}
  }

  for (auto a : A())
    for (auto b : A()) {
      __range.begin(); // expected-error {{use of undeclared identifier '__range'}}
      ++__begin; // expected-error {{use of undeclared identifier '__begin'}}
      --__end; // expected-error {{use of undeclared identifier '__end'}}
    }

  for (char c : "test")
    ;
  for (auto a : f()) // expected-error {{cannot use type 'void' as a range}}
    ;

  extern int incomplete[];
  for (auto a : incomplete) // expected-error {{cannot use incomplete type 'int []' as a range}}
    ;
  extern struct Incomplete also_incomplete[2]; // expected-note {{forward declaration}}
  for (auto &a : also_incomplete) // expected-error {{cannot use incomplete type 'struct Incomplete [2]' as a range}}
    ;

  struct VoidBegin {
    void begin(); // expected-note {{selected 'begin' function with iterator type 'void'}}
    void end();
  };
  for (auto a : VoidBegin()) // expected-error {{cannot use type 'void' as an iterator}}
    ;

  struct Differ {
    int *begin();
    null_t end();
  };
  for (auto a : Differ())
#if __cplusplus <= 201402L
    // expected-warning@-2 {{'begin' and 'end' returning different types ('int *' and 'null_t') is a C++1z extension}}
    // expected-note@-6 {{selected 'begin' function with iterator type 'int *'}}
    // expected-note@-6 {{selected 'end' function with iterator type 'null_t'}}
#endif
    ;

  for (void f() : "error") // expected-error {{for range declaration must declare a variable}}
    ;

  for (extern int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'extern'}}
  for (static int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'static'}}
  for (register int a : A()) {} // expected-error {{loop variable 'a' may not be declared 'register'}} expected-warning 0-1{{register}} expected-error 0-1{{register}}
  for (constexpr int a : X::C()) {} // OK per CWG issue #1204.

  for (auto u : X::NoBeginADL()) { // expected-error {{invalid range expression of type 'X::NoBeginADL'; no viable 'begin' function available}}
  }
  for (auto u : X::NoEndADL()) { // expected-error {{invalid range expression of type 'X::NoEndADL'; no viable 'end' function available}}
  }

  struct NoBegin {
    null_t end();
  };
  struct NoEnd {
    null_t begin();
  };
  for (auto u : NoBegin()) { // expected-error {{range type 'NoBegin' has 'end' member but no 'begin' member}}
  }
  for (auto u : NoEnd()) { // expected-error {{range type 'NoEnd' has 'begin' member but no 'end' member}} 
  }

  struct NoIncr {
    void *begin(); // expected-note {{selected 'begin' function with iterator type 'void *'}}
    void *end();
  };
  for (auto u : NoIncr()) { // expected-error {{arithmetic on a pointer to void}}\
    expected-note {{in implicit call to 'operator++' for iterator of type 'NoIncr'}}
  }

  struct NoNotEq {
    NoNotEq begin(); // expected-note {{selected 'begin' function with iterator type 'NoNotEq'}}
    NoNotEq end();
    void operator++();
  };
  for (auto u : NoNotEq()) { // expected-error {{invalid operands to binary expression}}\
    expected-note {{in implicit call to 'operator!=' for iterator of type 'NoNotEq'}}
  }

  struct NoDeref {
    NoDeref begin(); // expected-note {{selected 'begin' function}}
    NoDeref end();
    void operator++();
    bool operator!=(NoDeref &);
  };

  for (auto u : NoDeref()) { // expected-error {{indirection requires pointer operand}} \
    expected-note {{in implicit call to 'operator*' for iterator of type 'NoDeref'}}
  }

  struct NoCopy {
    NoCopy();
    NoCopy(const NoCopy &) = delete;
    int *begin();
    int *end();
  };
  for (int n : NoCopy()) { // ok
  }

  for (int n : 42) { // expected-error {{invalid range expression of type 'int'; no viable 'begin' function available}}
  }

  for (auto a : *also_incomplete) { // expected-error {{cannot use incomplete type 'struct Incomplete' as a range}}
  }
}

template<typename T, typename U>
void h(T t) {
  for (U u : t) { // expected-error {{no viable conversion from 'X::A' to 'int'}}
  }
  for (auto u : t) {
  }
}

template void h<A, int>(A);
template void h<A(&)[4], A &>(A(&)[4]);
template void h<A(&)[13], A>(A(&)[13]);
template void h<A(&)[13], int>(A(&)[13]); // expected-note {{requested here}}

template<typename T>
void i(T t) {
  for (auto u : t) { // expected-error {{invalid range expression of type 'X::A *'; no viable 'begin' function available}} \
                        expected-error {{member function 'begin' not viable}} \
                        expected-note {{when looking up 'begin' function}}

  }
}
template void i<A[13]>(A*); // expected-note {{requested here}}
template void i<const A>(const A); // expected-note {{requested here}}

struct StdBeginEnd {};
namespace std {
  int *begin(StdBeginEnd);
  int *end(StdBeginEnd);
}
void DR1442() {
  for (auto a : StdBeginEnd()) {} // expected-error {{invalid range expression of type 'StdBeginEnd'; no viable 'begin'}}
}

namespace NS {
  class ADL {};
  int *begin(ADL); // expected-note {{no known conversion from 'NS::NoADL' to 'NS::ADL'}}
  int *end(ADL);

  class NoADL {};
}
int *begin(NS::NoADL);
int *end(NS::NoADL);

struct VoidBeginADL {};
void begin(VoidBeginADL); // expected-note {{selected 'begin' function with iterator type 'void'}}
void end(VoidBeginADL);

void j() {
  for (auto u : NS::ADL()) {
  }
  for (auto u : NS::NoADL()) { // expected-error {{invalid range expression of type 'NS::NoADL'; no viable 'begin' function available}}
  }
  for (auto a : VoidBeginADL()) { // expected-error {{cannot use type 'void' as an iterator}}

  }
}

void example() {
  int array[5] = { 1, 2, 3, 4, 5 };
  for (int &x : array)
    x *= 2;
}

namespace rdar13712739 {
  template<typename T>
  void foo(const T& t) {
    auto &x = t.get(); // expected-error{{member reference base type 'const int' is not a structure or union}}
    for (auto &blah : x) { }
  }

  template void foo(const int&); // expected-note{{in instantiation of function template specialization}}
}
