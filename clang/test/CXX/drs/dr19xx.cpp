// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-11,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx98-11,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

namespace std { struct type_info; }

namespace dr1902 { // dr1902: 3.7
  struct A {};
  struct B {
    B(A); // #dr1902-B-A
    B() = delete; // #dr1902-B-ctor
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    B(const B&) = delete; // #dr1902-B-copy-ctor
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    operator A();
  };

  extern B b1;
  B b2(b1);
  // expected-error@-1 {{call to deleted constructor of 'B'}}
  //   expected-note@#dr1902-B-copy-ctor {{'B' has been explicitly marked deleted here}}

#if __cplusplus >= 201103L
  // This is ambiguous, even though calling the B(const B&) constructor would
  // both directly and indirectly call a deleted function.
  B b({});
  // since-cxx11-error@-1 {{call to constructor of 'B' is ambiguous}}
  //   since-cxx11-note@#dr1902-B-A {{candidate constructor}}
  //   since-cxx11-note@#dr1902-B-copy-ctor {{candidate constructor has been explicitly deleted}}
#endif
}

namespace dr1903 {
  namespace A {
    struct a {};
    int a;
    namespace B {
      int b;
    }
    using namespace B;
    namespace {
      int c;
    }
    namespace D {
      int d;
    }
    using D::d;
  }
  namespace X {
    using A::a;
    using A::b;
    using A::c;
    using A::d;
    struct a *p;
  }
}

namespace dr1909 { // dr1909: 3.7
  struct A {
    template<typename T> struct A {};
    // expected-error@-1 {{member 'A' has the same name as its class}}
  };
  struct B {
    template<typename T> void B() {}
    // expected-error@-1 {{constructor cannot have a return type}}
  };
  struct C {
    template<typename T> static int C;
    // expected-error@-1 {{member 'C' has the same name as its class}} 
    // cxx98-11-error@-2 {{variable templates are a C++14 extension}}
  };
  struct D {
    template<typename T> using D = int;
    // cxx98-error@-1 {{alias declarations are a C++11 extension}}
    // expected-error@-2 {{member 'D' has the same name as its class}}
  };
}

namespace dr1940 { // dr1940: 3.5
#if __cplusplus >= 201103L
static union {
  static_assert(true, "");  // ok
  static_assert(false, "");
  // since-cxx11-error@-1 {{static assertion failed}}
  int not_empty;
};
#endif
}

namespace dr1941 { // dr1941: 3.9
#if __cplusplus >= 201402L
template<typename X>
struct base {
  template<typename T>
  base(T a, T b, decltype(void(*T()), 0) = 0) {
    while (a != b) (void)*a++;
  }

  template<typename T>
  base(T a, X x, decltype(void(T(0) * 1), 0) = 0) {
    for (T n = 0; n != a; ++n) (void)X(x);
  }
};

struct derived : base<int> {
  using base::base;
};

struct iter {
  iter operator++(int);
  int operator*();
  friend bool operator!=(iter, iter);
} it, end;

derived d1(it, end);
derived d2(42, 9);
#endif
}

namespace dr1947 { // dr1947: 3.5
#if __cplusplus >= 201402L
unsigned o = 0'01;  // ok
unsigned b = 0b'01;
// since-cxx14-error@-1 {{invalid digit 'b' in octal constant}}
unsigned x = 0x'01;
// since-cxx14-error@-1 {{invalid suffix 'x'01' on integer constant}}
#endif
}

#if __cplusplus >= 201103L
// dr1948: 3.5
// FIXME: This diagnostic could be improved.
void *operator new(__SIZE_TYPE__) noexcept { return nullptr; }
// since-cxx11-error@-1 {{exception specification in declaration does not match previous declaration}}
#endif

namespace dr1959 { // dr1959: 3.9
#if __cplusplus >= 201103L
  struct b;
  struct c;
  struct a {
    a() = default;
    a(const a &) = delete; // #dr1959-copy-ctor
    a(const b &) = delete; // not inherited
    a(c &&) = delete; // #dr1959-move-ctor
    template<typename T> a(T) = delete; // #dr1959-temp-ctor
  };

  struct b : a { // #dr1959-b
    using a::a; // #dr1959-using-a
  };

  a x;
  // FIXME: As a resolution to an open DR against P0136R0, we disallow
  // use of inherited constructors to construct from a single argument
  // where the base class is reference-related to the argument type.
  b y = x;
  // since-cxx11-error@-1 {{no viable conversion from 'a' to 'b'}}
  //   since-cxx11-note@#dr1959-move-ctor {{candidate inherited constructor not viable: no known conversion from 'a' to 'c &&' for 1st argument}}
  //   since-cxx11-note@#dr1959-using-a {{constructor from base class 'a' inherited here}}
  //   since-cxx11-note@#dr1959-b {{candidate constructor (the implicit copy constructor) not viable: cannot bind base class object of type 'a' to derived class reference 'const b &' for 1st argument}}
  //   since-cxx11-note@#dr1959-temp-ctor {{candidate template ignored: instantiation would take its own class type by value}}
  //   since-cxx11-note@#dr1959-using-a {{constructor from base class 'a' inherited here}}
  b z = z;
  // since-cxx11-error@-1 {{call to implicitly-deleted copy constructor of 'b'}}
  //   since-cxx11-note@#dr1959-b {{copy constructor of 'b' is implicitly deleted because base class 'a' has a deleted copy constructor}}
  //   since-cxx11-note@#dr1959-copy-ctor {{'a' has been explicitly marked deleted here}}

  struct c : a {
    using a::a;
    c(const c &);
  };
  // FIXME: As a resolution to an open DR against P0136R0, we disallow
  // use of inherited constructors to construct from a single argument
  // where the base class is reference-related to the argument type.
  c q(static_cast<c&&>(q));
#endif
}

namespace dr1960 { // dr1960: no
struct A {
void f() {}
protected:
void g() {}
};

struct B: A {
private:
using A::f;
using A::g;
};

struct C : B {
// FIXME: both declarations are ill-formed, because A::f and A::g
// are not accessible.
using A::f;
using A::g;
};
}

namespace dr1966 { // dr1966: 11
#if __cplusplus >= 201103L
  struct A {
    enum E : int {1};
    // since-cxx11-error@-1 {{expected identifier}} (not bit-field)
  };
  auto *p1 = new enum E : int;
  // since-cxx11-error@-1 {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration}}
  auto *p2 = new enum F : int {};
  // since-cxx11-error@-1 {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration}}
  auto *p3 = true ? new enum G : int {};
  // since-cxx11-error@-1 {{ISO C++ forbids forward references to 'enum' types}}
  // since-cxx11-error@-2 {{allocation of incomplete type 'enum G'}}
  //   since-cxx11-note@-3 {{forward declaration of 'dr1966::G'}}
  auto h() -> enum E : int {};
  // since-cxx11-error@-1 {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration}}

  enum X : enum Y : int {} {};
  // since-cxx11-error@-1 {{'dr1966::Y' cannot be defined in a type specifier}}
  struct Q {
    // FIXME: can we emit something nicer than that?
    enum X : enum Y : int {} {};
    // since-cxx11-error@-1 {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
    // since-cxx11-error@-2 {{non-integral type 'enum Y' is an invalid underlying type}}
    // since-cxx11-error@-3 {{anonymous bit-field cannot have a default member initializer}}
  };
#endif
}

namespace dr1968 { // dr1968: no
#if __cplusplus >= 201103L
  // FIXME: According to DR1968, both of these should be considered
  // non-constant.
  static_assert(&typeid(int) == &typeid(int), "");

  constexpr const std::type_info *f() { return &typeid(int); }
  static_assert(f() == f(), "");
#endif
}

namespace dr1991 { // dr1991: 3.9
#if __cplusplus >= 201103L
  struct A {
    A(int, int) = delete;
  };

  struct B : A {
    using A::A;
    B(int, int, int = 0);
  };

  // FIXME: As a resolution to an open DR against P0136R1, we treat derived
  // class constructors as better than base class constructors in the presence
  // of ambiguity.
  B b(0, 0); // ok, calls B constructor
#endif
}

// dr1994: dup 529
