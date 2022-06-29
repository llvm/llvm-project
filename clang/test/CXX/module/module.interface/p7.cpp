// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/p7.cppm -emit-module-interface -o %t/p7.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify

//--- p7.cppm
export module p7;
struct reachable {
  constexpr static int sv = 43;
  int value = 44;

  static int getValue() { return 43; }
  int get() { return 44; }

  template <typename T>
  static bool templ_get(T t) { return false; }
  typedef int typedef_type;
  using using_type = int;
  template <typename T>
  using templ_using_type = int;
  bool operator()() {
    return false;
  }

  enum E { a,
           b };
};

export auto getReachable() {
  return reachable{};
}

export enum E1 { e1 };
enum E2 { e2 };
export using E2U = E2;
enum E3 { e3 };
export E3 func();

//--- Use.cpp
import p7;
void test() {
  auto reachable = getReachable();
  int a = decltype(reachable)::sv;
  int b = decltype(reachable)::getValue();
  int c = reachable.value;
  int d = reachable.get();
  int e = decltype(reachable)::a;
  int f = reachable.templ_get(a);
  typename decltype(reachable)::typedef_type g;
  typename decltype(reachable)::using_type h;
  typename decltype(reachable)::template templ_using_type<int> j;
  auto value = reachable();
}

void test2() {
  auto a = E1::e1;               // OK, namespace-scope name E1 is visible and e1 is reachable
  auto b = e1;                   // OK, namespace-scope name e1 is visible
  auto c = E2::e2;               // expected-error {{declaration of 'E2' must be imported from module}}
                                 // expected-note@* {{declaration here is not visible}}
  auto d = e2;                   // should be error, namespace-scope name e2 is not visible
  auto e = E2U::e2;              // OK, namespace-scope name E2U is visible and E2::e2 is reachable
  auto f = E3::e3;               // expected-error {{declaration of 'E3' must be imported from module 'p7' before it is required}}
                                 // expected-note@* {{declaration here is not visible}}
  auto g = e3;                   // should be error, namespace-scope name e3 is not visible
  auto h = decltype(func())::e3; // OK, namespace-scope name f is visible and E3::e3 is reachable
}
