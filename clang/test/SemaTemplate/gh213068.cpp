// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=c++17 -fsyntax-only -fdiagnostics-show-template-tree %s 2>&1 | FileCheck %s

template <template <template <typename> class, typename> class T,
          template <typename> class V>
struct PartialApply {
  template <template <template <typename> class, typename> class A,
            template <template <typename> class, typename> class B,
            template <typename> class F, typename X>
  using Mul = A<PartialApply<B, F>::template R, X>; // expected-note {{previous definition is here}}
  template <template <template <typename> class, typename> class T_ffl,
            template <typename> class V_ffl>
  struct PartialApply_ffl {};
  template <template <template <typename> class, typename> class A_ffl,
            template <template <typename> class, typename> class B_ffl,
            template <typename> class F_ffl, typename X>
  using Mul = // expected-error {{type alias template redefinition with different types}}
      A_ffl<PartialApply_ffl<B_ffl, F_ffl>::template R, X>;
};

// CHECK:      error: type alias template redefinition with different types
// CHECK:      [template PartialApply_ffl<B_ffl, F_ffl>::template R != template PartialApply<B, F>::template R],
// CHECK-NEXT: [...]
