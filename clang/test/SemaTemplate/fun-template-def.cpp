// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// Tests that dependent expressions are always allowed, whereas non-dependent
// are checked as usual.

#include <stddef.h>

// Fake typeid, lacking a typeinfo header.
namespace std { class type_info {}; }

struct dummy {}; // expected-note 3 {{candidate constructor (the implicit copy constructor)}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-2 3 {{candidate constructor (the implicit move constructor) not viable}}
#endif

template<typename T>
int f0(T x) {
  return (sizeof(x) == sizeof(int))? 0 : (sizeof(x) == sizeof(double))? 1 : 2;
}

template <typename T, typename U>
T f1(T t1, U u1, int i1, T** tpp)
{
  T t2 = i1;
  t2 = i1 + u1;
  ++u1;
  u1++;
  int i2 = u1;

  i1 = t1[u1];
  i1 *= t1;

  i1(u1, t1);
  u1(i1, t1);

  U u2 = (T)i1;
  static_cast<void>(static_cast<U>(reinterpret_cast<T>(
    dynamic_cast<U>(const_cast<T>(i1)))));

  new U(i1, t1);
  new int(t1, u1);
  new (t1, u1) int;
  delete t1;

  dummy d1 = sizeof(t1); // expected-error {{no viable conversion}}
  dummy d2 = offsetof(T, foo); // expected-error {{no viable conversion}}
  dummy d3 = __alignof(u1); // expected-error {{no viable conversion}}
  i1 = typeid(t1); // expected-error {{assigning to 'int' from incompatible type 'const std::type_info'}}
  i1 = tpp[0].size(); // expected-error {{'T *' is not a structure or union}}

  return u1;
}

template<typename T>
void f2(__restrict T x) {} // expected-note {{substitution failure [with T = int]: restrict requires a pointer or reference ('int' is invalid}}

void f3() {
  f2<int*>(0);
  f2<int>(0); // expected-error {{no matching function for call to 'f2'}}
}

#if __cplusplus >= 202002L
namespace GH138657 {
template <auto V> // #gh138657-template-head
class meta {};
template<int N>
class meta<N()> {}; // expected-error {{called object type 'int' is not a function or function point}}

template<int N[1]>
class meta<N()> {}; // expected-error {{called object type 'int *' is not a function or function point}}

template<char* N>
class meta<N()> {}; // expected-error {{called object type 'char *' is not a function or function point}}

struct S {};
template<S>
class meta<S()> {}; // expected-error {{template argument for non-type template parameter is treated as function type 'S ()'}}
                    // expected-note@#gh138657-template-head {{template parameter is declared here}}

}

namespace GH115725 {
template<auto ...> struct X {};
template<typename T, typename ...Ts> struct A {
  template<Ts ...Ns, T *...Ps>
  A(X<0(Ps)...>, Ts (*...qs)[Ns]);
  // expected-error@-1{{called object type 'int' is not a function or function pointer}}

};
}

namespace GH68852 {
template <auto v>
struct constexpr_value {
  template <class... Ts>
  constexpr constexpr_value<v(Ts::value...)> call(Ts...) {
    //expected-error@-1 {{called object type 'int' is not a function or function pointer}}
    return {};
  }
};

template <auto v> constexpr static inline auto c_ = constexpr_value<v>{};
// expected-note@-1 {{in instantiation of template}}
auto k = c_<1>; // expected-note {{in instantiation of variable}}

}

#endif
#if __cplusplus >= 201702L

namespace GH138731 {
template <class...>
using void_t = void;

template <class T>
T&& declval();

struct S {
  S();
  static int f();
  static int var;
};

namespace invoke_detail {

template <typename F>
struct traits {
  template <typename... A>
  using result = decltype(declval<F>()(declval<A>()...));
};

template <typename F, typename... A>
using invoke_result_t = typename traits<F>::template result<A...>;

template <typename Void, typename F, typename... A>
inline constexpr bool is_invocable_v = false;

template <typename F, typename... A>
inline constexpr bool
    is_invocable_v<void_t<invoke_result_t<F, A...>>, F, A...> = true;

}

template <typename F, typename... A>
inline constexpr bool is_invocable_v =
    invoke_detail::is_invocable_v<void, F, A...>;

static_assert(!is_invocable_v<int>);
static_assert(!is_invocable_v<int, int>);
static_assert(!is_invocable_v<S>);
static_assert(is_invocable_v<decltype(&S::f)>);
static_assert(!is_invocable_v<decltype(&S::var)>);

}

#endif
