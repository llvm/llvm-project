// RUN: %clang_cc1 -std=c++98 -pedantic-errors %s -verify=expected,cxx98
// RUN: %clang_cc1 -std=c++11 -pedantic-errors %s -verify=expected,since-cxx11,cxx11
// RUN: %clang_cc1 -std=c++14 -pedantic-errors %s -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++17 -pedantic-errors %s -verify=expected,since-cxx11
// RUN: %clang_cc1 -std=c++20 -pedantic-errors %s -verify=expected,since-cxx11,since-cxx20
// RUN: %clang_cc1 -std=c++23 -pedantic-errors %s -verify=expected,since-cxx11,since-cxx20,since-cxx23
// RUN: %clang_cc1 -std=c++2c -pedantic-errors %s -verify=expected,since-cxx11,since-cxx20,since-cxx23

#if __cplusplus == 199711L
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
// cxx98-error@-1 {{variadic macros are a C99 feature}}
#endif

namespace std {
#if __cplusplus >= 202002L
  struct strong_ordering {
    int n;
    constexpr operator int() const { return n; }
    static const strong_ordering less, equal, greater;
  };
  constexpr strong_ordering strong_ordering::less{-1},
      strong_ordering::equal{0}, strong_ordering::greater{1};
#endif

  typedef short int16_t;
  typedef unsigned short uint16_t;
  typedef int int32_t;
  typedef unsigned uint32_t;
  typedef long long int64_t;
  // cxx98-error@-1 {{'long long' is a C++11 extension}}
  typedef unsigned long long uint64_t;
  // cxx98-error@-1 {{'long long' is a C++11 extension}}
  static_assert(sizeof(int16_t) == 2 && sizeof(int32_t) == 4 && sizeof(int64_t) == 8, "Some tests rely on these sizes");

  template<typename T> T declval();
} // namespace std

namespace cwg2621 { // cwg2621: sup 2877
#if __cplusplus >= 202002L
enum class E { a };
namespace One {
using E_t = E;
using enum E_t; // typedef ok
auto v = a;
}
namespace Two {
using cwg2621::E;
int E; // ignored by type-only lookup
using enum E;
}
#endif
} // namespace cwg2621

namespace cwg2627 { // cwg2627: 20
#if __cplusplus >= 202002L
struct C {
  long long i : 8;
  friend auto operator<=>(C, C) = default;
};

void f() {
  C x{1}, y{2};
  static_cast<void>(x <=> y);
  static_cast<void>(x.i <=> y.i);
}

template<typename T>
struct CDependent {
  T i : 8;
  friend auto operator<=>(CDependent, CDependent) = default;
};

template<typename T>
concept three_way_comparable = requires(T t) { { t <=> t }; };
template<typename T>
concept bf_three_way_comparable = requires(T t) { { t.i <=> t.i }; };
static_assert(three_way_comparable<CDependent<long long>>);
static_assert(bf_three_way_comparable<CDependent<long long>>);
#endif

#if __cplusplus >= 201103L
template<typename T, int N>
struct D {
  T i : N;
};

template<typename T, int N>
D<T, N> d();

std::int32_t d1{ d<std::int64_t, 31>().i };
std::int32_t d2{ d<std::int64_t, 32>().i };
std::int32_t d3{ d<std::int64_t, 33>().i };
// since-cxx11-error@-1 {{non-constant-expression cannot be narrowed from type 'long long' to 'std::int32_t' (aka 'int') in initializer list}}
//   since-cxx11-note@-2 {{insert an explicit cast to silence this issue}}

std::int16_t d6{ d<int, 16>().i };
std::int16_t d7{ d<unsigned, 15>().i };
std::int16_t d8{ d<unsigned, 16>().i };
// since-cxx11-error@-1 {{non-constant-expression cannot be narrowed from type 'unsigned int' to 'std::int16_t' (aka 'short') in initializer list}}
//   since-cxx11-note@-2 {{insert an explicit cast to silence this issue}}
std::uint16_t d9{ d<unsigned, 16>().i };
std::uint16_t da{ d<int, 1>().i };
// since-cxx11-error@-1 {{non-constant-expression cannot be narrowed from type 'int' to 'std::uint16_t' (aka 'unsigned short') in initializer list}}
//   since-cxx11-note@-2 {{insert an explicit cast to silence this issue}}

bool db{ d<unsigned, 1>().i };
bool dc{ d<int, 1>().i };
// since-cxx11-error@-1 {{non-constant-expression cannot be narrowed from type 'int' to 'bool' in initializer list}}
//   since-cxx11-note@-2 {{insert an explicit cast to silence this issue}}

template<typename Target, typename Source>
constexpr decltype(Target{ std::declval<Source>().i }, false) is_narrowing(int) { return false; }
template<typename Target, typename Source>
constexpr bool is_narrowing(long) { return true; }

static_assert(!is_narrowing<std::int16_t, D<int, 16>>(0), "");
static_assert(!is_narrowing<std::int16_t, D<unsigned, 15>>(0), "");
static_assert(is_narrowing<std::int16_t, D<unsigned, 16>>(0), "");
static_assert(!is_narrowing<std::uint16_t, D<unsigned, 16>>(0), "");
static_assert(is_narrowing<std::uint16_t, D<int, 1>>(0), "");
static_assert(!is_narrowing<bool, D<unsigned, 1>>(0), "");
static_assert(is_narrowing<bool, D<int, 1>>(0), "");

template<int N>
struct E {
  signed int x : N;
  decltype(std::int16_t{ x }) dependent_narrowing;
  decltype(unsigned{ x }) always_narrowing;
  // since-cxx11-error@-1 {{non-constant-expression cannot be narrowed from type 'int' to 'unsigned int' in initializer list}}
  //   since-cxx11-note@-2 {{insert an explicit cast to silence this issue}}
};
#endif
} // namespace cwg2627

namespace cwg2628 { // cwg2628: 20
#if __cplusplus >= 202002L
template <bool A = false, bool B = false>
struct foo {
  constexpr foo() requires (!A && !B) = delete; // #cwg2628-ctor
  constexpr foo() requires (A || B) = delete;
};

void f() {
  foo fooable; // #cwg2628-fooable
  // since-cxx20-error@#cwg2628-fooable {{call to deleted}}
  //   since-cxx20-note@#cwg2628-ctor {{marked deleted here}}
}
#endif
} // namespace cwg2628

// cwg2630 is in cwg2630.cpp

namespace cwg2631 { // cwg2631: 16
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
} // namespace cwg2631

namespace cwg2635 { // cwg2635: 16
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
} // namespace cwg2635

// cwg2636: na

namespace cwg2640 { // cwg2640: 16

int \N{Λ} = 0;
// expected-error@-1 {{'Λ' is not a valid Unicode character name}}
// expected-error@-2 {{expected unqualified-id}}
const char* emoji = "\N{🤡}";
// expected-error@-1 {{'🤡' is not a valid Unicode character name}}
//   expected-note@-2 {{did you mean OX ('🐂' U+1F402)?}}
//   expected-note@-3 {{did you mean ANT ('🐜' U+1F41C)?}}
//   expected-note@-4 {{did you mean ARC ('⌒' U+2312)?}}
//   expected-note@-5 {{did you mean AXE ('🪓' U+1FA93)?}}
//   expected-note@-6 {{did you mean BAT ('🦇' U+1F987)?}}

#define z(x) 0
#define cwg2640_a z(
int x = cwg2640_a\N{abc});
// expected-error@-1 {{'abc' is not a valid Unicode character name}}
int y = cwg2640_a\N{LOTUS});
// expected-error@-1 {{character <U+1FAB7> not allowed in an identifier}}
// expected-error@-2 {{use of undeclared identifier 'cwg2640_a🪷'}}
} // namespace cwg2640

// cwg2642: na

namespace cwg2644 { // cwg2644: 8
#if __cplusplus >= 201103L
auto z = [a = 42](int a) {
// cxx11-error@-1 {{initialized lambda captures are a C++14 extension}}
// since-cxx11-error@-2 {{a lambda parameter cannot shadow an explicitly captured entity}}
//   since-cxx11-note@-3 {{variable 'a' is explicitly captured here}}
     return 1;
};
#endif
} // namespace cwg2644

namespace cwg2650 { // cwg2650: 17
#if __cplusplus >= 202302L
template <class T, T> struct S {};
template <class T> int f(S<T, T{}>*); // #cwg2650-f
class X {
  int m;
};
int i0 = f<X>(0);
// since-cxx23-error@-1 {{no matching function for call to 'f'}}
//   since-cxx23-note@#cwg2650-f {{type 'X' of non-type template parameter is not a structural type}}
#endif
} // namespace cwg2650

namespace cwg2653 { // cwg2653: 18
#if __cplusplus >= 202302L
  struct Test { void f(this const auto& = Test{}); };
  // since-cxx23-error@-1 {{the explicit object parameter cannot have a default argument}}
  auto L = [](this const auto& = Test{}){};
  // since-cxx23-error@-1 {{the explicit object parameter cannot have a default argument}}
#endif
} // namespace cwg2653

namespace cwg2654 { // cwg2654: 16
void f() {
    int neck, tail;
    volatile int brachiosaur;
    brachiosaur += neck;                // OK
    brachiosaur -= neck;                // OK
    brachiosaur |= neck;                // OK
}
} // namespace cwg2654

namespace cwg2681 { // cwg2681: 17
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
struct J { // #cwg2681-J
  unsigned char array[N];
};

H h = { "abc" };
I i = { "def" };
static_assert(__is_same(decltype(h), H<char, 4>));  // Not H<const char, 4>
static_assert(__is_same(decltype(i), I<char, 4>));

J j = { "ghi" };
// since-cxx20-error@-1 {{no viable constructor or deduction guide}}
//   since-cxx20-note@#cwg2681-J {{candidate template ignored: could not match 'cwg2681::J<N>' against 'const char *'}}
//   since-cxx20-note@#cwg2681-J {{implicit deduction guide declared as 'template <size_t N> J(cwg2681::J<N>) -> cwg2681::J<N>'}}
//   since-cxx20-note@#cwg2681-J {{candidate template ignored: could not match 'const unsigned char' against 'const char'}}
//   since-cxx20-note@#cwg2681-J {{implicit deduction guide declared as 'template <size_t N> J(const unsigned char (&)[N]) -> cwg2681::J<N>'}}
//   since-cxx20-note@#cwg2681-J {{candidate function template not viable: requires 0 arguments, but 1 was provided}}
//   since-cxx20-note@#cwg2681-J {{implicit deduction guide declared as 'template <size_t N> J() -> cwg2681::J<N>'}}
#endif
} // namespace cwg2681

namespace cwg2672 { // cwg2672: 18
#if __cplusplus >= 202002L
template <class T>
void f(T) requires requires { []() { T::invalid; } (); };
// since-cxx20-error@-1 {{type 'int' cannot be used prior to '::' because it has no members}}
//   since-cxx20-note@-2 {{while substituting into a lambda expression here}}
//   since-cxx20-note@-3 {{in instantiation of requirement here}}
//   since-cxx20-note@-4 {{while substituting template arguments into constraint expression here}}
//   since-cxx20-note@#cwg2672-f-0 {{while checking constraint satisfaction for template 'f<int>' required here}}
//   since-cxx20-note@#cwg2672-f-0 {{while substituting deduced template arguments into function template 'f' [with T = int]}}
void f(...);

template <class T>
void bar(T) requires requires {
   []() -> decltype(T::foo()) {};
};
void bar(...);

void m() {
  f(0); // #cwg2672-f-0
  bar(0);
}
#endif
} // namespace cwg2672

namespace cwg2687 { // cwg2687: 18
#if __cplusplus >= 202302L
struct S{
    void f(int);
    static void g(int);
    void h(this const S&, int);
};

void test() {
    (&S::f)(1);
    // since-cxx23-error@-1 {{called object type 'void (S::*)(int)' is not a function or function pointer}}
    (&S::g)(1);
    (&S::h)(S(), 1);
}
#endif
} // namespace cwg2687

namespace cwg2692 { // cwg2692: 19
#if __cplusplus >= 202302L

struct A {
    static void f(A); // #cwg2692-1
    void f(this A); // #cwg2692-2

    template <typename T>
    static void g(T); // #cwg2692-3
    template <typename T>
    void g(this T); // #cwg2692-4

    void test();
};

void A::test() {
    (&A::f)(A());
    // since-cxx23-error@-1 {{call to 'f' is ambiguous}}
    //   since-cxx23-note@#cwg2692-1 {{candidate function}}
    //   since-cxx23-note@#cwg2692-2 {{candidate function}}
    (&A::f)();
    // since-cxx23-error@-1 {{no matching function for call to 'f'}}
    //   since-cxx23-note@#cwg2692-1 {{candidate function not viable: requires 1 argument, but 0 were provided}}
    //   since-cxx23-note@#cwg2692-2 {{candidate function not viable: requires 1 argument, but 0 were provided}}


    (&A::g)(A());
    // since-cxx23-error@-1 {{call to 'g' is ambiguous}}
    //   since-cxx23-note@#cwg2692-3 {{candidate function}}
    //   since-cxx23-note@#cwg2692-4 {{candidate function}}
    (&A::g<A>)();
    // since-cxx23-error@-1 {{no matching function for call to 'g'}}
    //   since-cxx23-note@#cwg2692-3 {{candidate function template not viable: requires 1 argument, but 0 were provided}}
    //   since-cxx23-note@#cwg2692-4 {{candidate function [with T = cwg2692::A] not viable: requires 1 argument, but 0 were provided}}
}
#endif
} // namespace cwg2692
