// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface -verify %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cpp -fmodule-file=A=%t/A.pcm -fsyntax-only -verify

//--- A.cppm
export module A;
static void f() {}
inline void it() { f(); }           // expected-warning {{TU local entity 'f' is exposed}}
static inline void its() { f(); }   // OK
template<int> void g() { its(); }   // OK
template void g<0>();

// Developers Note: We didn't track the use in decltype(). But it should be fine
// since the underlying type is not TU-local. So we're doing nothing bad in practice.
decltype(f) *fp;                    // error: f (though not its type) is TU-local
auto &fr = f;                       // OK
// Developers Note: We didn't track the use across variables. In the current implementation,
// we don't know the value of `fr` at compile time, so we failed to detect this.
constexpr auto &fr2 = fr;           // error: is an exposure of f
// Developers Note: But if it is a direct use, we are able to detect it.
constexpr auto &fr3 = f;            // expected-warning {{TU local entity 'f' is exposed}}
constexpr static auto fp2 = fr;     // OK

struct S { void (&ref)(); } s{f};               // OK, value is TU-local
constexpr extern struct W { S &s; } wrap{s};    // OK, value is not TU-local

static auto x = []{f();};           // OK
auto x2 = x;                        // expected-warning {{TU local entity}}
// Developers Note: Why is this an exposure?
int y = ([]{f();}(),0);             // error: the closure type is not TU-local
int y2 = (x,0);                     // OK expected-warning{{left operand of comma operator has no effect}}

namespace N {
  struct A {};
  void adl(A);
  static void adl(int);
}
void adl(double);

inline void h(auto x) { adl(x); }   // OK, but certain specializations are exposures

// Reflection is not supported yet.
// constexpr std::meta::info r1 = ^^g<0>;  // OK
// namespace N2 {
//   static constexpr std::meta::info r2 = ^^g<1>;     // OK, r2 is TU-local
// }
// constexpr std::meta::info r3 = ^^f;                 // error: r3 is an exposure of f
// 
// constexpr auto ctx = std::meta::access_context::current();
// constexpr std::meta::info r4 =
//   std::meta::members_of(^^N2, ctx)[0];              // error: r4 is an exposure of N2​::​r2

//--- A.cpp
module A;
void other() {
  g<0>();                           // OK, specialization is explicitly instantiated
  g<1>();                           // expected-warning {{instantiation of 'g<1>' triggers reference to TU-local entity 'its' from other TU 'A'}}
  // Developers Note: To check use of TU-local entity when overload resolution made.
  h(N::A{});                        // error: overload set contains TU-local N​::​adl(int)
  h(0);                             // OK, calls adl(double)
  adl(N::A{});                      // OK; N​::​adl(int) not found, calls N​::​adl(N​::​A)
  fr();                             // OK, calls f
  // Developers Note: To check use of TU-local entity when we're able to detect the TUlocalness
  // across variables.
  constexpr auto ptr = fr;          // error: fr is not usable in constant expressions here

  constexpr auto fptr = f;          // expected-error {{use of undeclared identifier 'f'}}
}
