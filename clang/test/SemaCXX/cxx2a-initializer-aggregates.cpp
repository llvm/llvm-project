// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,pedantic,override,reorder -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,pedantic,override,reorder -Wno-c++20-designator -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,pedantic -Werror=c99-designator -Wno-reorder-init-list -Wno-initializer-overrides -Werror=nested-anon-types -Werror=gnu-anonymous-struct
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,reorder -Wno-c99-designator -Werror=reorder-init-list -Wno-initializer-overrides
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,override -Wno-c99-designator -Wno-reorder-init-list -Werror=initializer-overrides
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected -Wno-c99-designator -Wno-reorder-init-list -Wno-initializer-overrides
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,wmissing -Wmissing-field-initializers -Wno-c99-designator -Wno-reorder-init-list -Wno-initializer-overrides


namespace class_with_ctor {
  struct A { // cxx20-note 6{{candidate}}
    A() = default; // cxx20-note 3{{candidate}}
    int x;
    int y;
  };
  A a = {1, 2}; // cxx20-error {{no matching constructor}}

  struct B {
    int x;
    int y;
  };
  B b1 = B(); // trigger declaration of implicit ctors
  B b2 = {1, 2}; // ok

  struct C : A {
    A a;
  };
  C c1 = {{}, {}}; // ok, call default ctor twice
  C c2 = {{1, 2}, {3, 4}}; // cxx20-error 2{{no matching constructor}}
}

namespace designator {
struct A { int x, y; };
struct B { A a; };

A a1 = {
  .y = 1, // reorder-note {{previous initialization for field 'y' is here}}
  .x = 2 // reorder-error {{ISO C++ requires field designators to be specified in declaration order; field 'y' will be initialized after field 'x'}}
};
int arr[3] = {[1] = 5}; // pedantic-error {{array designators are a C99 extension}}
B b = {.a.x = 0}; // pedantic-error {{nested designators are a C99 extension}}
                  // wmissing-warning@-1 {{missing field 'y' initializer}}
A a2 = {
  .x = 1, // pedantic-error {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
  2 // pedantic-note {{first non-designated initializer is here}}
};
A a3 = {
  1, // pedantic-note {{first non-designated initializer is here}}
  .y = 2 // pedantic-error {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
};
A a4 = {
  .x = 1, // override-note {{previous}}
  .x = 1 // override-error {{overrides prior initialization}}
}; // wmissing-warning {{missing field 'y' initializer}}
A a5 = {
  .y = 1, // override-note {{previous}}
  .y = 1 // override-error {{overrides prior initialization}}
}; // wmissing-warning {{missing field 'x' initializer}}
B b2 = {.a = 1}; // pedantic-error {{brace elision for designated initializer is a C99 extension}}
                 // wmissing-warning@-1 {{missing field 'y' initializer}}
B b3 = {.a = 1, 2}; // pedantic-error {{mixture of designated and non-designated}} pedantic-note {{first non-designated}} pedantic-error {{brace elision}}
B b4 = {.a = 1, 2, 3}; // pedantic-error {{mixture of designated and non-designated}} pedantic-note {{first non-designated}} pedantic-error {{brace elision}} expected-error {{excess elements}}
B b5 = {.a = nullptr}; // expected-error {{cannot initialize}}
struct C { int :0, x, :0, y, :0; };
C c = {
  .x = 1, // override-note {{previous}}
  .x = 1, // override-error {{overrides prior initialization}} override-note {{previous}}
  .y = 1, // override-note {{previous}}
  .y = 1, // override-error {{overrides prior initialization}} // reorder-note {{previous initialization for field 'y' is here}}
  .x = 1, // reorder-error {{declaration order}} override-error {{overrides prior initialization}} override-note {{previous}}
  .x = 1, // override-error {{overrides prior initialization}}
};

struct Foo { int a, b; };

struct Foo foo0 = { 1 }; // wmissing-warning {{missing field 'b' initializer}}
struct Foo foo1 = { .a = 1 }; // wmissing-warning {{missing field 'b' initializer}}
struct Foo foo2 = { .b = 1 }; // wmissing-warning {{missing field 'a' initializer}}

}

namespace base_class {
  struct base {
    int x;
  };
  struct derived : base {
    int y;
  };
  derived d = {.x = 1, .y = 2}; // expected-error {{'x' does not refer to any field}}
}

namespace union_ {
  union U { int a, b; };
  U u = {
    .a = 1, // override-note {{here}}
    .b = 2, // override-error {{overrides prior}}
  };
}

namespace overload_resolution {
  struct A { int x, y; };
  union B { int x, y; };

  void f(A a);
  void f(B b) = delete;
  void g() { f({.x = 1, .y = 2}); } // ok, calls non-union overload

  // As an extension of the union case, overload resolution won't pick any
  // candidate where a field initializer would be overridden.
  struct A2 { int x, other, y; };
  int f(A2);
  void g2() { int k = f({.x = 1, 2, .y = 3}); (void)k; } // pedantic-error {{mixture of designated and non-designated}} pedantic-note {{here}}

  struct C { int x; };
  void h(A a); // expected-note {{candidate}}
  void h(C c); // expected-note {{candidate}}
  void i() {
    h({.x = 1, .y = 2});
    h({.y = 1, .x = 2}); // reorder-error {{declaration order}} reorder-note {{previous}}
    h({.x = 1}); // expected-error {{ambiguous}}
  }

  struct D { int y, x; };
  void j(A a); // expected-note {{candidate}}
  void j(D d); // expected-note {{candidate}}
  void k() {
    j({.x = 1, .y = 2}); // expected-error {{ambiguous}}
  }

  struct E { A a; };
  struct F { int a; };
  void l(E e); // expected-note {{candidate}}
  int &l(F f); // expected-note {{candidate}}
  void m() {
    int &r = l({.a = 0}); // ok, l(E) is not viable
    int &s = l({.a = {0}}); // expected-error {{ambiguous}}
  }
}

namespace deduction {
  struct A { int x, y; };
  union B { int x, y; };

  template<typename T, typename U> void f(decltype(T{.x = 1, .y = 2}) = {});
  template<typename T, typename U> void f(decltype(U{.x = 1, .y = 2}) = {}) = delete;
  void g() { f<A, B>(); } // ok, calls non-union overload

  struct C { int y, x; };
  template<typename T, typename U> void h(decltype(T{.y = 1, .x = 2}) = {}) = delete;
  template<typename T, typename U> void h(decltype(U{.y = 1, .x = 2}) = {});
  void i() {
    h<A, C>(); // ok, selects C overload by SFINAE
  }

  struct D { int n; };
  struct E { D n; };
  template<typename T, typename U> void j1(decltype(T{.n = 0}));
  template<typename T, typename U> void j1(decltype(U{.n = 0})) = delete;
  template<typename T, typename U> void j2(decltype(T{.n = {0}})); // expected-note {{candidate}}
  template<typename T, typename U> void j2(decltype(U{.n = {0}})); // expected-note {{candidate}}
  template<typename T, typename U> void j3(decltype(T{.n = {{0}}})) = delete;
  template<typename T, typename U> void j3(decltype(U{.n = {{0}}}));
  void k() {
    j1<D, E>({}); // ok, selects D overload by SFINAE (too few braces for E)
    j2<D, E>({}); // expected-error {{ambiguous}}
    j3<D, E>({}); // ok, selects E overload by SFINAE (too many braces for D)
  }
}

namespace no_unwrap {
  template<typename T> struct X {
    static_assert(false, "should not be instantiated");
  };
  struct Q {
    template<typename T, typename U = typename X<T>::type> Q(T&&);
  };

  // Ensure that we do not try to call 'Q::Q(.a = 1)' here.
  void g() { Q q = {.a = 1}; } // expected-error {{initialization of non-aggregate type 'Q' with a designated initializer list}}

  struct S { int a; };
  void h(Q q);
  void h(S s);

  // OK, does not instantiate X<void&> (!).
  void i() {
    h({.a = 1});
  }
}

namespace GH63605 {
struct A  {
  unsigned x;
  unsigned y;
  unsigned z;
};

struct B {
  unsigned a;
  unsigned b;
};

struct : public A, public B {
  unsigned : 2;
  unsigned a : 6;
  unsigned : 1;
  unsigned b : 6;
  unsigned : 2;
  unsigned c : 6;
  unsigned d : 1;
  unsigned e : 2;
} data = {
  {.z=0,
         // pedantic-note@-1 {{first non-designated initializer is here}}
         // reorder-note@-2 {{previous initialization for field 'z' is here}}
   .y=1, // reorder-error {{field 'z' will be initialized after field 'y'}}
         // reorder-note@-1 {{previous initialization for field 'y' is here}}
   .x=2}, // reorder-error {{field 'y' will be initialized after field 'x'}}
  {.b=3,  // reorder-note {{previous initialization for field 'b' is here}}
   .a=4}, // reorder-error {{field 'b' will be initialized after field 'a'}}
    .e = 1, // reorder-note {{previous initialization for field 'e' is here}}
            // pedantic-error@-1 {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
    .d = 1, // reorder-error {{field 'e' will be initialized after field 'd'}}
            // reorder-note@-1 {{previous initialization for field 'd' is here}}
    .c = 1, // reorder-error {{field 'd' will be initialized after field 'c'}} // reorder-note {{previous initialization for field 'c' is here}}
    .b = 1, // reorder-error {{field 'c' will be initialized after field 'b'}} // reorder-note {{previous initialization for field 'b' is here}}
    .a = 1, // reorder-error {{field 'b' will be initialized after field 'a'}}
};
}

namespace GH63759 {
struct C {
  int y = 1;
  union {
    int a;
    short b;
  };
  int x = 1;
};

void foo() {
  C c1 = {.x = 3, .a = 1}; // reorder-error-re {{ISO C++ requires field designators to be specified in declaration order; field 'x' will be initialized after field 'GH63759::C::(anonymous union at {{.*}})'}}
                           // reorder-note@-1 {{previous initialization for field 'x' is here}}

  C c2 = {.a = 3, .y = 1}; // reorder-error-re {{ISO C++ requires field designators to be specified in declaration order; field 'GH63759::C::(anonymous union at {{.*}})' will be initialized after field 'y'}}
                           // reorder-note-re@-1 {{previous initialization for field 'GH63759::C::(anonymous union at {{.*}})' is here}}
                           //
}
}

namespace GH70384 {

struct A {
  int m;
  union { int a; float n = 0; };
};

struct B {
  int m;
  int b;
  union { int a ; };
};

union CU {
  int a = 1;
  double b;
};

struct C {
  int a;
  union { int b; CU c;};
};

struct CC {
  int a;
  CU c;
};

void foo() {
  A a = A{.m = 0};
  A aa = {0};
  A aaa = {.a = 7}; // wmissing-warning {{missing field 'm' initializer}}
  B b = {.m = 1, .b = 3 }; //wmissing-warning {{missing field 'a' initializer}}
  B bb = {1}; // wmissing-warning {{missing field 'b' initializer}}
              // wmissing-warning@-1 {{missing field 'a' initializer}}
  C c = {.a = 1}; // wmissing-warning {{missing field 'b' initializer}}
  CC cc = {.a = 1}; // wmissing-warning {{missing field 'c' initializer}}
}

struct C1 {
  int m;
  union { float b; union {int n = 1; }; };
  // pedantic-error@-1 {{anonymous types declared in an anonymous union are an extension}}
};

struct C2 {
  int m;
  struct { float b; int n = 1; }; // pedantic-error {{anonymous structs are a GNU extension}}
};

struct C3 {
  int m;
  struct { float b = 1; union {int a;}; int n = 1; };
  // pedantic-error@-1 {{anonymous structs are a GNU extension}}
  // pedantic-error@-2 {{anonymous types declared in an anonymous struct are an extension}}
};

C1 c = C1{.m = 1};
C1 cc = C1{.b = 1}; // wmissing-warning {{missing field 'm' initializer}}
C2 c1 = C2{.m = 1}; // wmissing-warning {{missing field 'b' initializer}}
C2 c22 = C2{.m = 1, .b = 1};
C3 c2 = C3{.b = 1}; // wmissing-warning {{missing field 'a' initializer}}
                    // wmissing-warning@-1 {{missing field 'm' initializer}}

struct C4 {
  union {
    struct { int n; }; // pedantic-error {{anonymous structs are a GNU extension}}
    // pedantic-error@-1 {{anonymous types declared in an anonymous union are an extension}}
    int m = 0; };
  int z;
};
C4 a = {.z = 1};

struct C5 {
  int a;
  struct { // pedantic-error {{anonymous structs are a GNU extension}}
    int x;
    struct { int y = 0; };  // pedantic-error {{anonymous types declared in an anonymous struct are an extension}}
                            // pedantic-error@-1 {{anonymous structs are a GNU extension}}
  };
};
C5 c5 = C5{.a = 0}; //wmissing-warning {{missing field 'x' initializer}}
}
