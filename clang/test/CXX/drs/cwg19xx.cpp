// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98-11,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx98-11,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx14,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

namespace std {
struct type_info;
} // namespace std

namespace cwg1900 { // cwg1900: 2.7
// See the test for CWG1477 for detailed analysis
namespace N {
struct A {
  friend int f();
};
}
int N::f() { return 0; }
int N::g() { return 0; } 
// expected-error@-1 {{out-of-line definition of 'g' does not match any declaration in namespace 'cwg1900::N'}}
} // namespace cwg1900

namespace cwg1902 { // cwg1902: 3.7
  struct A {};
  struct B {
    B(A); // #cwg1902-B-A
    B() = delete; // #cwg1902-B-ctor
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    B(const B&) = delete; // #cwg1902-B-copy-ctor
    // cxx98-error@-1 {{deleted function definitions are a C++11 extension}}
    operator A();
  };

  extern B b1;
  B b2(b1);
  // expected-error@-1 {{call to deleted constructor of 'B'}}
  //   expected-note@#cwg1902-B-copy-ctor {{'B' has been explicitly marked deleted here}}

#if __cplusplus >= 201103L
  // This is ambiguous, even though calling the B(const B&) constructor would
  // both directly and indirectly call a deleted function.
  B b({});
  // since-cxx11-error@-1 {{call to constructor of 'B' is ambiguous}}
  //   since-cxx11-note@#cwg1902-B-A {{candidate constructor}}
  //   since-cxx11-note@#cwg1902-B-copy-ctor {{candidate constructor has been explicitly deleted}}
#endif
} // namespace cwg1902

namespace cwg1903 { // cwg1903: 2.7
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
} // namespace cwg1903

namespace cwg1909 { // cwg1909: 3.7
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
} // namespace cwg1909

namespace cwg1918 { // cwg1918: no
template<typename T> struct A {
  class B {
    class C {};
  };
};
class X {
  static int x;
  // FIXME: this is ill-formed, because A<T>::B::C does not end with a simple-template-id
  template <typename T>
  friend class A<T>::B::C;
  // expected-warning@-1 {{dependent nested name specifier 'A<T>::B::' for friend class declaration is not supported; turning off access control for 'X'}}
};
template<> struct A<int> {
  typedef struct Q B;
};
struct Q {
  class C {
    // FIXME: 'f' is not a friend, so 'X::x' is not accessible
    int f() { return X::x; }
  };
};
} // namespace cwg1918

namespace cwg1940 { // cwg1940: 3.5
#if __cplusplus >= 201103L
static union {
  static_assert(true, "");  // ok
  static_assert(false, "");
  // since-cxx11-error@-1 {{static assertion failed}}
  int not_empty;
};
#endif
} // namespace cwg1918

namespace cwg1941 { // cwg1941: 3.9
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
} // namespace cwg1941

namespace cwg1945 { // cwg1945: no
template<typename T> struct A {
  class B {
    class C {};
  };
};
class X {
  static int x;
  // FIXME: this is ill-formed, because A<T>::B::C does not end with a simple-template-id
  template <typename T>
  friend class A<T>::B::C;
  // expected-warning@-1 {{dependent nested name specifier 'A<T>::B::' for friend class declaration is not supported; turning off access control for 'X'}}
};
} // namespace cwg1945

namespace cwg1947 { // cwg1947: 3.5
#if __cplusplus >= 201402L
unsigned o = 0'01;  // ok
unsigned b = 0b'01;
// since-cxx14-error@-1 {{invalid digit 'b' in octal constant}}
unsigned x = 0x'01;
// since-cxx14-error@-1 {{invalid suffix 'x'01' on integer constant}}
#endif
} // namespace cwg1947

#if __cplusplus >= 201103L
// cwg1948: 3.5
// FIXME: This diagnostic could be improved.
void *operator new(__SIZE_TYPE__) noexcept { return nullptr; }
// since-cxx11-error@-1 {{exception specification in declaration does not match previous declaration}}
#endif

namespace cwg1959 { // cwg1959: 3.9
#if __cplusplus >= 201103L
  struct b;
  struct c;
  struct a {
    a() = default;
    a(const a &) = delete; // #cwg1959-copy-ctor
    a(const b &) = delete; // not inherited
    a(c &&) = delete; // #cwg1959-move-ctor
    template<typename T> a(T) = delete; // #cwg1959-temp-ctor
  };

  struct b : a { // #cwg1959-b
    using a::a; // #cwg1959-using-a
  };

  a x;
  // FIXME: As a resolution to an open DR against P0136R0, we disallow
  // use of inherited constructors to construct from a single argument
  // where the base class is reference-related to the argument type.
  b y = x;
  // since-cxx11-error@-1 {{no viable conversion from 'a' to 'b'}}
  //   since-cxx11-note@#cwg1959-move-ctor {{candidate inherited constructor not viable: no known conversion from 'a' to 'c &&' for 1st argument}}
  //   since-cxx11-note@#cwg1959-using-a {{constructor from base class 'a' inherited here}}
  //   since-cxx11-note@#cwg1959-b {{candidate constructor (the implicit copy constructor) not viable: cannot bind base class object of type 'a' to derived class reference 'const b &' for 1st argument}}
  //   since-cxx11-note@#cwg1959-temp-ctor {{candidate template ignored: instantiation would take its own class type by value}}
  //   since-cxx11-note@#cwg1959-using-a {{constructor from base class 'a' inherited here}}
  b z = z;
  // since-cxx11-error@-1 {{call to implicitly-deleted copy constructor of 'b'}}
  //   since-cxx11-note@#cwg1959-b {{copy constructor of 'b' is implicitly deleted because base class 'a' has a deleted copy constructor}}
  //   since-cxx11-note@#cwg1959-copy-ctor {{'a' has been explicitly marked deleted here}}

  struct c : a {
    using a::a;
    c(const c &);
  };
  // FIXME: As a resolution to an open DR against P0136R0, we disallow
  // use of inherited constructors to construct from a single argument
  // where the base class is reference-related to the argument type.
  c q(static_cast<c&&>(q));
#endif
} // namespace cwg1959

namespace cwg1960 { // cwg1960: no
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
} // namespace cwg1960

namespace cwg1966 { // cwg1966: 11
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
  //   since-cxx11-note@-3 {{forward declaration of 'cwg1966::G'}}
  auto h() -> enum E : int {};
  // since-cxx11-error@-1 {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration}}

  enum X : enum Y : int {} {};
  // since-cxx11-error@-1 {{'cwg1966::Y' cannot be defined in a type specifier}}
  struct Q {
    // FIXME: can we emit something nicer than that?
    enum X : enum Y : int {} {};
    // since-cxx11-error@-1 {{non-defining declaration of enumeration with a fixed underlying type is only permitted as a standalone declaration; missing list of enumerators?}}
    // since-cxx11-error@-2 {{non-integral type 'enum Y' is an invalid underlying type}}
    // since-cxx11-error@-3 {{anonymous bit-field cannot have a default member initializer}}
  };
#endif
} // namespace cwg1966

namespace cwg1968 { // cwg1968: no
#if __cplusplus >= 201103L
  // FIXME: According to CWG1968, both of these should be considered
  // non-constant.
  static_assert(&typeid(int) == &typeid(int), "");

  constexpr const std::type_info *f() { return &typeid(int); }
  static_assert(f() == f(), "");
#endif
} // namespace cwg1968

namespace cwg1991 { // cwg1991: 3.9
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
} // namespace cwg1991

// cwg1994: dup 529
