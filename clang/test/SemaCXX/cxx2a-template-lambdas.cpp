// RUN: %clang_cc1 -std=c++03 -verify -Dstatic_assert=_Static_assert -Wno-c++11-extensions -Wno-c++14-extensions -Wno-c++17-extensions -Wno-c++20-extensions %s
// RUN: %clang_cc1 -std=c++11 -verify=expected,cxx11,cxx11-cxx14 -Wno-c++20-extensions -Wno-c++17-extensions -Wno-c++14-extensions  %s
// RUN: %clang_cc1 -std=c++14 -verify=expected,cxx11-cxx14,cxx14 -Wno-c++20-extensions -Wno-c++17-extensions %s
// RUN: %clang_cc1 -std=c++17 -verify -Wno-c++20-extensions %s
// RUN: %clang_cc1 -std=c++20 -verify %s

template<typename, typename>
inline const bool is_same = false;

template<typename T>
inline const bool is_same<T, T> = true;

template<typename T>
struct DummyTemplate { };

void func() {
  auto L0 = []<typename T>(T arg) {
    static_assert(is_same<T, int>); // expected-error {{static assertion failed}}
  };
  L0(0);
  L0(0.0); // expected-note {{in instantiation}}

  auto L1 = []<int I> {
    static_assert(I == 5); // expected-error {{static assertion failed}}
  };
  L1.operator()<5>();
  L1.operator()<6>(); // expected-note {{in instantiation}}

  auto L2 = []<template<typename> class T, class U>(T<U> &&arg) {
    static_assert(is_same<T<U>, DummyTemplate<float> >); // // expected-error {{static assertion failed}}
  };
  L2(DummyTemplate<float>());
  L2(DummyTemplate<double>()); // expected-note {{in instantiation}}
}

template<typename T> // expected-note {{declared here}}
struct ShadowMe {
  void member_func() {
    auto L = []<typename T> { }; // expected-error {{'T' shadows template parameter}}
  }
};

#if __cplusplus >= 201102L
template<typename T>
constexpr T outer() {
  // FIXME: The C++11 error seems wrong
  return []<T x>() { return x; }.template operator()<123>(); // expected-error {{no matching member function}}  \
                                                                expected-note {{candidate template ignored}}    \
        cxx11-note {{non-literal type '<dependent type>' cannot be used in a constant expression}} \
        cxx14-note {{non-literal type}}
}
static_assert(outer<int>() == 123); // cxx11-cxx14-error {{not an integral constant expression}} cxx11-cxx14-note {{in call}}
template int *outer<int *>(); // expected-note {{in instantiation}}
#endif

#if __cplusplus >= 202002L
namespace GH62611 {
template <auto A = [](auto x){}>
struct C {
  static constexpr auto B = A;
};

int test() {
  C<>::B(42);
}

namespace AutoParam
{
template <auto A = [](auto x) { return x;}>
auto B = A;
static_assert(B<>(42) == 42);
}

namespace TypeParam
{
template <typename T = decltype([](auto x) {return x;})>
auto B = T{};
static_assert(B<>(42) == 42);
}

}

namespace GH64689 {
void f();
void foo() {
  []<typename T>(int)
    noexcept(requires(int t) { f(); })
    -> decltype(requires(int t) { f(); })
    requires requires(int t) { f(); }
  {return {};}.operator()<int>(0);
  [](auto)
    noexcept(requires(int t) { f(); })
    -> decltype(requires(int t) { f(); })
    requires requires(int t) { f(); }
  {return {};}(1);
}

}
#endif

#if __cplusplus >= 202002L
namespace {
struct S {};
constexpr S gs;
void f() {
  constexpr int x{};
  const int y{};
  auto b = []<int=x, int=y>{};
  using A = decltype([]<int=x>{});

  int z; // expected-note {{'z' declared here}}
  auto c = []<int t=z>{
    // expected-error@-1 {{no matching function for call to object of type}} \
    // expected-error@-1 {{variable 'z' cannot be implicitly captured in a lambda with no capture-default specified}} \
    // expected-note@-1 {{lambda expression begins here}} \
    // expected-note@-1 4{{capture}} \
    // expected-note@-1 {{candidate template ignored: substitution failure: reference to local variable 'z' declared in enclosing function}}
    return t;
  }();

  auto class_type_global = []<S=gs>{};

  static constexpr S static_s;
  auto class_type_static = []<S=static_s>{};

  constexpr S s;  // expected-note {{'s' declared here}}
  auto class_type = []<S=s>{};
  // expected-error@-1 {{variable 's' cannot be implicitly captured in a lambda with no capture-default specified}} \
  // expected-note@-1 {{lambda expression begins here}} \
  // expected-note@-1 4{{capture}}
}
}
#endif
