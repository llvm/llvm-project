// Tests that the definition in private module fragment is not reachable to its users.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Private.cppm -emit-module-interface -o %t/Private.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only

//--- Private.cppm
export module Private;
inline void fn_m(); // OK, module-linkage inline function
static void fn_s();
export struct X;

export void g(X *x) {
  fn_s(); // OK, call to static function in same translation unit
  fn_m(); // OK, call to module-linkage inline function
}
export X *factory(); // OK

module :private;
struct X {}; // definition not reachable from importers of A
X *factory() {
  return new X();
}
void fn_m() {}
void fn_s() {}

//--- Use.cpp
import Private;
void foo() {
  X x; // expected-error {{definition of 'X' must be imported from module 'Private.<private>' before it is required}}
       // expected-error@-1 {{definition of 'X' must be imported from module 'Private.<private>' before it is required}}
       // expected-note@* {{definition here is not reachable}}
       // expected-note@* {{definition here is not reachable}}
  auto _ = factory();
  auto *__ = factory();
  X *___ = factory();

  g(__);
  g(___);
  g(factory());
}
