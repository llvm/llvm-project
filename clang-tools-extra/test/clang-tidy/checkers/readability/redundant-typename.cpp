// RUN: %check_clang_tidy -std=c++11,c++14,c++17 %s readability-redundant-typename %t \
// RUN:   -- -- -fno-delayed-template-parsing
// RUN: %check_clang_tidy -std=c++20-or-later -check-suffixes=,20 %s readability-redundant-typename %t \
// RUN:   -- -- -fno-delayed-template-parsing

struct NotDependent {
  using R = int;
  struct S {};
};

auto f(typename NotDependent::S)
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES: auto f(NotDependent::S)
  -> typename NotDependent::R
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES: -> NotDependent::R
{
  return typename NotDependent::R();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant 'typename' [readability-redundant-typename]
  // return NotDependent::R();
}

template <
  typename T,
  typename T::R V,
  // CHECK-MESSAGES-20: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: T::R V,
  typename U = typename T::R
  // CHECK-MESSAGES-20: :[[@LINE-1]]:16: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: typename U = T::R
>
auto f() -> typename T::R
// CHECK-MESSAGES-20: :[[@LINE-1]]:13: warning: redundant 'typename' [readability-redundant-typename]
// CHECK-FIXES-20: auto f() -> T::R
{
  static_cast<typename T::R>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:15: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: static_cast<T::R>(0);

  dynamic_cast<typename T::R>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:16: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: dynamic_cast<T::R>(0);

  reinterpret_cast<typename T::R>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:20: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: reinterpret_cast<T::R>(0);

  const_cast<typename T::R>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:14: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: const_cast<T::R>(0);

  static_cast<typename T::R&>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:15: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: static_cast<T::R&>(0);

  dynamic_cast<typename T::R const volatile &&>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:16: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: dynamic_cast<T::R const volatile &&>(0);

  reinterpret_cast<const typename T::template M<42>::R *>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:26: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: reinterpret_cast<const T::template M<42>::R *>(0);

  const_cast<const typename T::R *const[100]>(0);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:20: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: const_cast<const T::R *const[100]>(0);

  (typename T::R)(0);

  alignof(typename T::R);

  new typename T::R();
  // CHECK-MESSAGES-20: :[[@LINE-1]]:7: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: new T::R();

  // CHECK-MESSAGES-20: :[[@LINE+2]]:15: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: static_cast<decltype([] {
  static_cast<typename decltype([] {
    return typename T::R(); // Inner typename must stay.
  })::R>(0);

  auto localFunctionDeclaration() -> typename T::R;
  // CHECK-MESSAGES-20: :[[@LINE-1]]:38: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: auto localFunctionDeclaration() -> T::R;

  void (*PointerToFunction)(typename T::R);
  void anotherLocalFunctionDeclaration(typename T::R);

  typename T::R DependentVar;
  typename NotDependent::R NotDependentVar;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES: NotDependent::R NotDependentVar;

  return typename T::R();
}

template <typename T>
using trait = const typename T::R ****;
// CHECK-MESSAGES-20: :[[@LINE-1]]:21: warning: redundant 'typename' [readability-redundant-typename]
// CHECK-FIXES-20: using trait = const T::R ****;

template <typename T>
using t = typename T::template R<T>;
// CHECK-MESSAGES-20: :[[@LINE-1]]:11: warning: redundant 'typename' [readability-redundant-typename]
// CHECK-FIXES-20: using t = T::template R<T>;

template <typename T>
trait<typename T::R> m();

#if __cplusplus >= 202002L

template <typename T>
concept c = requires(typename T::R) {
// CHECK-MESSAGES-20: :[[@LINE-1]]:22: warning: redundant 'typename' [readability-redundant-typename]
// CHECK-FIXES-20: concept c = requires(T::R) {
  typename T::R;
};

template <typename T>
requires c<typename T::R>
void b();

#endif // __cplusplus >= 202002L

template <typename T, typename>
struct PartiallySpecializedType {};

template <typename T>
struct PartiallySpecializedType<T, typename T::R> {};

#if __cplusplus >= 201402L

template <typename T>
typename T::R v = typename T::R();
// CHECK-MESSAGES-20: :[[@LINE-1]]:1: warning: redundant 'typename' [readability-redundant-typename]
// CHECK-FIXES-20: T::R v = typename T::R();

#endif // __cplusplus >= 201402L

template <typename T>
typename T::R f();
// CHECK-MESSAGES-20: :[[@LINE-1]]:1: warning: redundant 'typename' [readability-redundant-typename]
// CHECK-FIXES-20: T::R f();

template <typename T>
void n(typename T::R *) {}

template void n<NotDependent>(NotDependent::R *);

namespace ns {

template <typename T>
void f(typename T::R1, typename T::R2);

} // namespace ns

template <typename T>
void ns::f(
  typename T::R1,
  // CHECK-MESSAGES-20: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: T::R1,
  typename T::R2
  // CHECK-MESSAGES-20: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: T::R2
);

template <typename... Ts>
void p(typename Ts::R...);

template <typename T, typename... Ts>
class A {
public:
  friend typename T::R;
  // CHECK-MESSAGES-20: :[[@LINE-1]]:10: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: friend T::R;

  typedef typename T::R a;
  // CHECK-MESSAGES-20: :[[@LINE-1]]:11: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: typedef T::R a;

  const typename T::R typedef b;
  // CHECK-MESSAGES-20: :[[@LINE-1]]:9: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: const T::R typedef b;

  typename T::R v;
  // CHECK-MESSAGES-20: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: T::R v;

  typename T::R
  // CHECK-MESSAGES-20: :[[@LINE-1]]:3: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: T::R
  g(typename T::R) {}
  // CHECK-MESSAGES-20: :[[@LINE-1]]:5: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: g(T::R) {}

  void h(typename T::R = typename T::R()) {}
  // CHECK-MESSAGES-20: :[[@LINE-1]]:10: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: void h(T::R = typename T::R()) {}

  void p(typename Ts::R...);
  // CHECK-MESSAGES-20: :[[@LINE-1]]:10: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: void p(Ts::R...);

  friend void k(typename T::R) {}
  // CHECK-MESSAGES-20: :[[@LINE-1]]:17: warning: redundant 'typename' [readability-redundant-typename]
  // CHECK-FIXES-20: friend void k(T::R) {}

  friend struct T::R;
  using typename T::R;
  enum E1 : typename T::R {};
  enum class E2 : typename T::R {};
  operator typename T::R();
  void m() { this->operator typename T::R(); }
};
