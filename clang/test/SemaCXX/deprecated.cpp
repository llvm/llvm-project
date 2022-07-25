// RUN: %clang_cc1 -std=c++98 %s -Wno-parentheses -Wdeprecated -verify=expected,not-cxx20 -triple x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++11 %s -Wno-parentheses -Wdeprecated -verify=expected,not-cxx20 -triple x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++14 %s -Wno-parentheses -Wdeprecated -verify=expected,not-cxx20 -triple x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++17 %s -Wno-parentheses -Wdeprecated -verify=expected,not-cxx20 -triple x86_64-linux-gnu
// RUN: %clang_cc1 -std=c++2a %s -Wno-parentheses -Wdeprecated -verify=expected,cxx20 -triple x86_64-linux-gnu

// RUN: %clang_cc1 -std=c++14 %s -Wno-parentheses -Wdeprecated -verify=expected,not-cxx20 -triple x86_64-linux-gnu -Wno-deprecated-register -DNO_DEPRECATED_FLAGS

#include "Inputs/register.h"

namespace std {
  struct type_info {};
}

void g() throw();
void h() throw(int);
void i() throw(...);
#if __cplusplus > 201402L
// expected-warning@-4 {{dynamic exception specifications are deprecated}} expected-note@-4 {{use 'noexcept' instead}}
// expected-error@-4 {{ISO C++17 does not allow dynamic exception specifications}} expected-note@-4 {{use 'noexcept(false)' instead}}
// expected-error@-4 {{ISO C++17 does not allow dynamic exception specifications}} expected-note@-4 {{use 'noexcept(false)' instead}}
#elif __cplusplus >= 201103L
// expected-warning@-8 {{dynamic exception specifications are deprecated}} expected-note@-8 {{use 'noexcept' instead}}
// expected-warning@-8 {{dynamic exception specifications are deprecated}} expected-note@-8 {{use 'noexcept(false)' instead}}
// expected-warning@-8 {{dynamic exception specifications are deprecated}} expected-note@-8 {{use 'noexcept(false)' instead}}
#endif

void stuff(register int q) {
#if __cplusplus > 201402L
  // expected-error@-2 {{ISO C++17 does not allow 'register' storage class specifier}}
#elif __cplusplus >= 201103L && !defined(NO_DEPRECATED_FLAGS)
  // expected-warning@-4 {{'register' storage class specifier is deprecated}}
#endif
  register int n;
#if __cplusplus > 201402L
  // expected-error@-2 {{ISO C++17 does not allow 'register' storage class specifier}}
#elif __cplusplus >= 201103L && !defined(NO_DEPRECATED_FLAGS)
  // expected-warning@-4 {{'register' storage class specifier is deprecated}}
#endif

  register int m asm("rbx"); // no-warning

  int k = to_int(n); // no-warning
  bool b;
  ++b;
#if __cplusplus > 201402L
  // expected-error@-2 {{ISO C++17 does not allow incrementing expression of type bool}}
#else
  // expected-warning@-4 {{incrementing expression of type bool is deprecated}}
#endif

  b++;
#if __cplusplus > 201402L
  // expected-error@-2 {{ISO C++17 does not allow incrementing expression of type bool}}
#else
  // expected-warning@-4 {{incrementing expression of type bool is deprecated}}
#endif

  char *p = "foo";
#if __cplusplus < 201103L
  // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#else
  // expected-warning@-4 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#endif
}

struct S { int n; void operator+(int); };
struct T : private S {
  S::n;
#if __cplusplus < 201103L
  // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
  // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif
  S::operator+;
#if __cplusplus < 201103L
  // expected-warning@-2 {{access declarations are deprecated; use using declarations instead}}
#else
  // expected-error@-4 {{ISO C++11 does not allow access declarations; use using declarations instead}}
#endif
};

#if __cplusplus >= 201103L
namespace DeprecatedCopy {
  struct Assign {
    Assign &operator=(const Assign&); // expected-warning {{definition of implicit copy constructor for 'Assign' is deprecated because it has a user-declared copy assignment operator}}
  };
  Assign a1, a2(a1); // expected-note {{implicit copy constructor for 'DeprecatedCopy::Assign' first required here}}

  struct Ctor {
    Ctor();
    Ctor(const Ctor&); // expected-warning {{definition of implicit copy assignment operator for 'Ctor' is deprecated because it has a user-declared copy constructor}}
  };
  Ctor b1, b2;
  void f() { b1 = b2; } // expected-note {{implicit copy assignment operator for 'DeprecatedCopy::Ctor' first required here}}

  struct Dtor {
    ~Dtor();
    // expected-warning@-1 {{definition of implicit copy constructor for 'Dtor' is deprecated because it has a user-declared destructor}}
    // expected-warning@-2 {{definition of implicit copy assignment operator for 'Dtor' is deprecated because it has a user-declared destructor}}
  };
  Dtor c1, c2(c1); // expected-note {{implicit copy constructor for 'DeprecatedCopy::Dtor' first required here}}
  void g() { c1 = c2; } // expected-note {{implicit copy assignment operator for 'DeprecatedCopy::Dtor' first required here}}

  struct DefaultedDtor {
    ~DefaultedDtor() = default;
  };
  DefaultedDtor d1, d2(d1);
  void h() { d1 = d2; }
}
#endif

struct X {
  friend int operator,(X, X);
  void operator[](int);
};
void array_index_comma() {
  int arr[123];
  (void)arr[(void)1, 2];
  (void)arr[X(), X()];
  X()[(void)1, 2];
  X()[X(), X()];
#if __cplusplus > 201703L
  // expected-warning@-5 {{deprecated}}
  // expected-warning@-5 {{deprecated}}
  // expected-warning@-5 {{deprecated}}
  // expected-warning@-5 {{deprecated}}
#endif

  (void)arr[((void)1, 2)];
  (void)arr[(X(), X())];
  (void)((void)1,2)[arr];
  (void)(X(), X())[arr];
  X()[((void)1, 2)];
  X()[(X(), X())];
}

namespace DeprecatedVolatile {
  volatile int n = 1;
  void use(int);
  void f() {
    // simple assignments are deprecated only if their value is used
    n = 5; // ok
#if __cplusplus >= 201103L
    decltype(n = 5) m = n; // ok expected-warning {{side effects}}
    (void)noexcept(n = 5); // ok expected-warning {{side effects}}
#endif
    (void)typeid(n = 5); // ok expected-warning {{side effects}}
    (n = 5, 0); // ok
    use(n = 5); // cxx20-warning {{use of result of assignment to object of volatile-qualified type 'volatile int' is deprecated}}
    int q = n = 5; // cxx20-warning {{deprecated}}
    q = n = 5; // cxx20-warning {{deprecated}}
#if __cplusplus >= 201103L
    decltype(q = n = 5) m2 = q; // cxx20-warning {{deprecated}} expected-warning {{side effects}}
    (void)noexcept(q = n = 5); // cxx20-warning {{deprecated}} expected-warning {{side effects}}
#endif
    (void)sizeof(q = n = 5); // cxx20-warning {{deprecated}} expected-warning {{side effects}}
    (void)typeid(use(n = 5)); // cxx20-warning {{deprecated}} expected-warning {{side effects}}
    (void)__alignof(+(n = 5)); // cxx20-warning {{deprecated}} expected-warning {{side effects}}

    // FIXME: These cases are technically deprecated because the parens are
    // part of the operand, but we choose to not diagnose for now.
    (void)sizeof(n = 5); // expected-warning {{side effects}}
    (void)__alignof(n = 5); // expected-warning {{side effects}}
    // Similarly here.
    (n = 5);

    volatile bool b = true;
    if (b = true) {} // cxx20-warning {{deprecated}}
    for (b = true;
         b = true; // cxx20-warning {{deprecated}}
         b = true) {}
    for (volatile bool x = true;
         volatile bool y = true; // ok despite volatile load from volatile initialization
        ) {}

    // inc / dec / compound assignments are always deprecated
    ++n; // cxx20-warning {{increment of object of volatile-qualified type 'volatile int' is deprecated}}
    --n; // cxx20-warning {{decrement of object of volatile-qualified type 'volatile int' is deprecated}}
    n++; // cxx20-warning {{increment of object of volatile-qualified type 'volatile int' is deprecated}}
    n--; // cxx20-warning {{decrement of object of volatile-qualified type 'volatile int' is deprecated}}
    n += 5; // cxx20-warning {{compound assignment to object of volatile-qualified type 'volatile int' is deprecated}}
    n *= 3; // cxx20-warning {{compound assignment to object of volatile-qualified type 'volatile int' is deprecated}}
    n /= 2; // cxx20-warning {{compound assignment to object of volatile-qualified type 'volatile int' is deprecated}}
    n %= 42; // cxx20-warning {{compound assignment to object of volatile-qualified type 'volatile int' is deprecated}}
    n &= 2; // undeprecated as a DR in C++23
    n |= 2; // undeprecated as a DR in C++23
    n ^= 2; // undeprecated as a DR in C++23

    (void)__is_trivially_assignable(volatile int&, int); // no warning

#if __cplusplus >= 201703L
    struct X { int a, b; };
    volatile auto [x, y] = X{1, 2}; // cxx20-warning {{volatile qualifier in structured binding declaration is deprecated}}

    struct Y { volatile int a, b; };
    auto [x2, y2] = Y{1, 2}; // ok
#endif
  }
  volatile int g( // cxx20-warning {{volatile-qualified return type 'volatile int' is deprecated}}
      volatile int n, // cxx20-warning {{volatile-qualified parameter type 'volatile int' is deprecated}}
      volatile int (*p)( // cxx20-warning {{volatile-qualified return type 'volatile int' is deprecated}}
        volatile int m) // cxx20-warning {{volatile-qualified parameter type 'volatile int' is deprecated}}
      );
#if __cplusplus >= 201103L
  auto lambda = []( // cxx20-warning{{volatile-qualified return type 'volatile int' is deprecated}}
      volatile int n) // cxx20-warning{{volatile-qualified parameter type 'volatile int' is deprecated}}
    -> volatile int { return n; };
#endif

  template<typename T> T f(T v); // cxx20-warning 2{{deprecated}}
  int use_f = f<volatile int>(0); // FIXME: Missing "in instantiation of" note.

  // OK, only the built-in operators are deprecated.
  struct UDT {
    UDT(volatile const UDT&);
    UDT &operator=(const UDT&);
    UDT &operator=(const UDT&) volatile;
    UDT operator+=(const UDT&) volatile;
  };
  void h(UDT a) {
    volatile UDT b = a;
    volatile UDT c = b;
    a = c = a;
    b += a;
  }
}

namespace ArithConv {
  enum E { e } e2;
  enum F { f };
  bool b1 = e == e2;
  bool b2 = e == f; // not-cxx20-warning-re {{different enumeration types ('ArithConv::E' and 'ArithConv::F'){{$}}}} cxx20-warning {{F') is deprecated}}
  bool b3 = e == 0.0; // cxx20-warning {{comparison of enumeration type 'ArithConv::E' with floating-point type 'double' is deprecated}}
  bool b4 = 0.0 == f; // cxx20-warning {{comparison of floating-point type 'double' with enumeration type 'ArithConv::F' is deprecated}}
  int n1 = true ? e : f; // cxx20-warning {{conditional expression between different enumeration types ('ArithConv::E' and 'ArithConv::F') is deprecated}}
  int n2 = true ? e : 0.0; // cxx20-warning {{conditional expression between enumeration type 'ArithConv::E' and floating-point type 'double' is deprecated}}
}

namespace ArrayComp {
  int arr1[3], arr2[4];
  bool b1 = arr1 == arr2; // expected-warning {{array comparison always evaluates to false}} cxx20-warning {{comparison between two arrays is deprecated}}
  bool b2 = arr1 < arr2; // expected-warning {{array comparison always evaluates to a constant}} cxx20-warning {{comparison between two arrays is deprecated}}
  __attribute__((weak)) int arr3[3];
  bool b3 = arr1 == arr3; // cxx20-warning {{comparison between two arrays is deprecated}}
  bool b4 = arr1 < arr3; // cxx20-warning {{comparison between two arrays is deprecated}}
#if __cplusplus > 201703L
  bool b5 = arr1 <=> arr2; // cxx20-error {{invalid operands}}
#endif

  int (&f())[3];
  bool b6 = arr1 == f(); // cxx20-warning {{comparison between two arrays is deprecated}}
  bool b7 = arr1 == +f();
}

# 1 "/usr/include/system-header.h" 1 3
void system_header_function(void) throw();
