// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,cxx11
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20,since-cxx23
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected,since-cxx11,since-cxx20,since-cxx23


namespace dr2621 { // dr2621: 16
#if __cplusplus >= 202002L
enum class E { a };
namespace One {
using E_t = E;
using enum E_t; // typedef ok
auto v = a;
}
namespace Two {
using dr2621::E;
int E; // we see this
using enum E;
// since-cxx20-error@-1 {{unknown type name E}}
}
#endif
}

namespace dr2628 { // dr2628: no open
                   // this was reverted for the 16.x release
                   // due to regressions, see the issue for more details:
                   // https://github.com/llvm/llvm-project/issues/60777
#if __cplusplus >= 202002L
template <bool A = false, bool B = false>
struct foo {
  // The expected notes below should be removed when dr2628 is fully implemented again
  constexpr foo() requires (!A && !B) = delete; // #dr2628-ctor-1
  constexpr foo() requires (A || B) = delete; //  #dr2628-ctor-2
};

void f() {
  // The FIXME's below should be the expected errors when dr2628 is
  // fully implemented again.
  foo fooable; // #dr2628-fooable
  // since-cxx20-error@-1 {{ambiguous deduction for template arguments of 'foo'}}
  //   since-cxx20-note@#dr2628-ctor-1 {{candidate function [with A = false, B = false]}}
  //   since-cxx20-note@#dr2628-ctor-2 {{candidate function [with A = false, B = false]}}
  // FIXME-since-cxx20-error@#dr2628-fooable {{call to deleted}} 
  //   FIXME-since-cxx20-note@#dr2628-ctor {{marked deleted here}} 
}
#endif
}

namespace dr2631 { // dr2631: 16
#if __cplusplus >= 202002L
  constexpr int g();
  consteval int f() {
    return g();
  }
  int k(int x = f()) {
    return x;
  }
  constexpr int g() {
    return 42;
  }
  int test() {
    return k();
  }
#endif
}

namespace dr2635 { // dr2635: 16
#if __cplusplus >= 202002L
template<typename T>
concept UnaryC = true;
template<typename T, typename U>
concept BinaryC = true;

struct S{ int i, j; };
S get_S();

template<typename T>
T get_T();

void use() {
  UnaryC auto [a, b] = get_S();
  // since-cxx20-error@-1 {{decomposition declaration cannot be declared with constrained 'auto'}}
  BinaryC<int> auto [c, d] = get_S();
  // since-cxx20-error@-1 {{decomposition declaration cannot be declared with constrained 'auto'}}
}

template<typename T>
void TemplUse() {
  UnaryC auto [a, b] = get_T<T>();
  // since-cxx20-error@-1 {{decomposition declaration cannot be declared with constrained 'auto'}}
  BinaryC<T> auto [c, d] = get_T<T>();
  // since-cxx20-error@-1 {{decomposition declaration cannot be declared with constrained 'auto'}}
}
#endif
}

// dr2636: na

namespace dr2640 { // dr2640: 16

int \N{Λ} = 0;
// expected-error@-1 {{'Λ' is not a valid Unicode character name}}
// expected-error@-2 {{expected unqualified-id}}
const char* emoji = "\N{🤡}";
// expected-error@-1 {{'🤡' is not a valid Unicode character name}}
// expected-note@-2 {{did you mean OX ('🐂' U+1F402)?}}
// expected-note@-3 {{did you mean ANT ('🐜' U+1F41C)?}}
// expected-note@-4 {{did you mean ARC ('⌒' U+2312)?}}
// expected-note@-5 {{did you mean AXE ('🪓' U+1FA93)?}}
// expected-note@-6 {{did you mean BAT ('🦇' U+1F987)?}}

#define z(x) 0
#define dr2640_a z(
int x = dr2640_a\N{abc});
// expected-error@-1 {{'abc' is not a valid Unicode character name}}
int y = dr2640_a\N{LOTUS});
// expected-error@-1 {{character <U+1FAB7> not allowed in an identifier}}
// expected-error@-2 {{use of undeclared identifier 'dr2640_a🪷'}}
// expected-error@-3 {{extraneous ')' before ';'}}
}

// dr2642: na

namespace dr2644 { // dr2644: 8
#if __cplusplus >= 201103L
auto z = [a = 42](int a) {
// cxx11-warning@-1 {{initialized lambda captures are a C++14 extension}}
// since-cxx11-error@-2 {{a lambda parameter cannot shadow an explicitly captured entity}}
//   since-cxx11-note@-3 {{variable 'a' is explicitly captured here}}
     return 1;
};
#endif
}

#if __cplusplus >= 202302L
namespace dr2650 { // dr2650: 17
template <class T, T> struct S {};
template <class T> int f(S<T, T{}>*); // #dr2650-f
class X {
  int m;
};
int i0 = f<X>(0);
// since-cxx23-error@-1 {{no matching function for call to 'f'}}
//   since-cxx23-note@#dr2650-f {{type 'X' of non-type template parameter is not a structural type}}
}
#endif

#if __cplusplus >= 202302L
namespace dr2653 { // dr2653: 18
  struct Test { void f(this const auto& = Test{}); };
  // since-cxx23-error@-1 {{the explicit object parameter cannot have a default argument}}
  auto L = [](this const auto& = Test{}){};
  // since-cxx23-error@-1 {{the explicit object parameter cannot have a default argument}}
}
#endif

namespace dr2654 { // dr2654: 16
void f() {
    int neck, tail;
    volatile int brachiosaur;
    brachiosaur += neck;                // OK
    brachiosaur -= neck;                // OK
    brachiosaur |= neck;                // OK
}
}

namespace dr2681 { // dr2681: 17
#if __cplusplus >= 202002L
using size_t = decltype(sizeof(int));

template<class T, size_t N>
struct H {
  T array[N];
};
template<class T, size_t N>
struct I {
  volatile T array[N];
};
template<size_t N>
struct J { // #dr2681-J
  unsigned char array[N];
};

H h = { "abc" };
I i = { "def" };
static_assert(__is_same(decltype(h), H<char, 4>));  // Not H<const char, 4>
static_assert(__is_same(decltype(i), I<char, 4>));

J j = { "ghi" };
// since-cxx20-error@-1 {{no viable constructor or deduction guide}}
//   since-cxx20-note@#dr2681-J {{candidate template ignored: could not match 'J<N>' against 'const char *'}}
//   since-cxx20-note@#dr2681-J {{candidate template ignored: could not match 'const unsigned char' against 'const char'}}
//   since-cxx20-note@#dr2681-J {{candidate function template not viable: requires 0 arguments, but 1 was provided}}
#endif
}

namespace dr2672 { // dr2672: 18 open
#if __cplusplus >= 202002L
template <class T>
void f(T) requires requires { []() { T::invalid; } (); };
// since-cxx20-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
//   since-cxx20-note@-2 {{while substituting into a lambda expression here}}
//   since-cxx20-note@-3 {{in instantiation of requirement here}}
//   since-cxx20-note@-4 {{while substituting template arguments into constraint expression here}}
//   since-cxx20-note@#dr2672-f-0 {{while checking constraint satisfaction for template 'f<int>' required here}}
//   since-cxx20-note@#dr2672-f-0 {{in instantiation of function template specialization 'dr2672::f<int>' requested here}}
void f(...);

template <class T>
void bar(T) requires requires {
   decltype([]() -> T {})::foo();
};
void bar(...);

void m() {
  f(0); // #dr2672-f-0
  bar(0);
}
#endif
}

#if __cplusplus >= 202302L
namespace dr2687 { // dr2687: 18
struct S{
    void f(int);
    static void g(int);
    void h(this const S&, int);
};

void test() {
    (&S::f)(1);
    // since-cxx23-error@-1 {{called object type 'void (dr2687::S::*)(int)' is not a function or function pointer}}
    (&S::g)(1);
    (&S::h)(S(), 1);
}
}
#endif
