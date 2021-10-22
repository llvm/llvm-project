// RUN: %clang_cc1 -std=c++0x -fsyntax-only -verify %s

struct one { char c[1]; };
struct two { char c[2]; };

namespace aggregate {
  struct S {
    int ar[2];
    struct T {
      int i1;
      int i2;
    } t;
    struct U {
      int i1;
    } u[2];
    struct V {
      int var[2];
    } v;
  };

  void bracing() {
    S s1 = { 1, 2, 3 ,4, 5, 6, 7, 8 };
    S s2{ {1, 2}, {3, 4}, { {5}, {6} }, { {7, 8} } };
    S s3{ 1, 2, 3, 4, 5, 6 };
    S s4{ {1, 2}, {3, 4}, {5, 6}, { {7, 8} } };
    S s5{ {1, 2}, {3, 4}, { {5}, {6} }, {7, 8} };
  }

  void bracing_new() {
    new S{ {1, 2}, {3, 4}, { {5}, {6} }, { {7, 8} } };
    new S{ 1, 2, 3, 4, 5, 6 };
    new S{ {1, 2}, {3, 4}, {5, 6}, { {7, 8} } };
    new S{ {1, 2}, {3, 4}, { {5}, {6} }, {7, 8} };
  }

  void bracing_construct() {
    (void) S{ {1, 2}, {3, 4}, { {5}, {6} }, { {7, 8} } };
    (void) S{ 1, 2, 3, 4, 5, 6 };
    (void) S{ {1, 2}, {3, 4}, {5, 6}, { {7, 8} } };
    (void) S{ {1, 2}, {3, 4}, { {5}, {6} }, {7, 8} };
  }

  struct String {
    String(const char*);
  };

  struct A {
    int m1;
    int m2;
  };

  void function_call() {
    void takes_A(A);
    takes_A({1, 2});
  }

  struct B {
    int m1;
    String m2;
  };

  void overloaded_call() {
    one overloaded(A);
    two overloaded(B);

    static_assert(sizeof(overloaded({1, 2})) == sizeof(one), "bad overload");
    static_assert(sizeof(overloaded({1, "two"})) == sizeof(two),
      "bad overload");
    // String is not default-constructible
    static_assert(sizeof(overloaded({1})) == sizeof(one), "bad overload");
  }

  struct C { int a[2]; C():a({1, 2}) { } }; // expected-error {{parenthesized initialization of a member array is a GNU extension}}
}

namespace array_explicit_conversion {
  typedef int test1[2];
  typedef int test2[];
  template<int x> struct A { int a[x]; }; // expected-error {{'a' declared as an array with a negative size}}
  typedef A<1> test3[];
  typedef A<-1> test4[];
  void f() {
    (void)test1{1};
    (void)test2{1};
    (void)test3{{{1}}};
    (void)test4{{{1}}}; // expected-note {{in instantiation of template class 'array_explicit_conversion::A<-1>' requested here}}
  }
}

namespace sub_constructor {
  struct DefaultConstructor { // expected-note 2 {{not viable}}
    DefaultConstructor(); // expected-note  {{not viable}}
    int x;
  };
  struct NoDefaultConstructor1 { // expected-note 2 {{not viable}}
    NoDefaultConstructor1(int); // expected-note {{not viable}}
    int x;
  };
  struct NoDefaultConstructor2 {  // expected-note 4 {{not viable}}
    NoDefaultConstructor2(int,int); // expected-note 2 {{not viable}}
    int x;
  };

  struct Aggr {
    DefaultConstructor a;
    NoDefaultConstructor1 b;
    NoDefaultConstructor2 c;
  };

  Aggr ok1 { {}, {0} , {0,0} };
  Aggr ok2 = { {}, {0} , {0,0} };
  Aggr too_many { {0} , {0} , {0,0} }; // expected-error {{no matching constructor for initialization}}
  Aggr too_few { {} , {0} , {0} }; // expected-error {{no matching constructor for initialization}}
  Aggr invalid { {} , {&ok1} , {0,0} }; // expected-error {{no matching constructor for initialization}}
  NoDefaultConstructor2 array_ok[] = { {0,0} , {0,1} };
  NoDefaultConstructor2 array_error[] = { {0,0} , {0} }; // expected-error {{no matching constructor for initialization}}
}

namespace multidimensional_array {
  void g(const int (&)[2][2]) {}
  void g(const int (&)[2][2][2]) = delete;

  void h() {
    g({{1,2},{3,4}});
  }
}

namespace array_addressof {
  using T = int[5];
  T *p = &T{1,2,3,4,5}; // expected-error {{taking the address of a temporary object of type 'array_addressof::T' (aka 'int[5]')}}
}

namespace PR24816 {
  struct { int i; } ne = {{0, 1}}; // expected-error{{excess elements in scalar initializer}}
}

namespace no_crash {
class Foo; // expected-note {{forward declaration}}
void test(int size) {
  Foo array[size] = {0}; // expected-error {{variable has incomplete type}}
}
}
