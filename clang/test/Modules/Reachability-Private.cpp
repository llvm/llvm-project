// Tests that the definition in private module fragment is not reachable to its users.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Private.cppm -emit-module-interface \
// RUN: -o %t/Private.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp \
// RUN: -DTEST_BADINLINE -verify -fsyntax-only

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Private.cppm -emit-reduced-module-interface \
// RUN: -o %t/Private.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp \
// RUN: -DTEST_BADINLINE -verify -fsyntax-only

//--- Private.cppm
export module Private;
#ifdef TEST_BADINLINE
inline void fn_m(); // expected-error {{un-exported inline function not defined before the private module fragment}}
                    // expected-note@Private.cppm:13 {{private module fragment begins here}}
#endif
static void fn_s();
export struct X;

export void g(X *x) {
  fn_s(); // OK, call to static function in same translation unit
#ifdef TEST_BADINLINE
  fn_m(); // fn_m is not OK.
#endif
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
  X x; // expected-error 1+{{missing '#include'; 'X' must be defined before it is used}}
       // expected-note@Private.cppm:18 1+{{definition here is not reachable}}
  auto _ = factory();
  auto *__ = factory();
  X *___ = factory();

  g(__);
  g(___);
  g(factory());
}
