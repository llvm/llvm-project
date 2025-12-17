// Note: the run lines follow their respective tests, since line/column
// matter in this test.

template <int...> struct B {};
template <int> class C;

namespace method {
  struct S {
    template <int Z>
    void waldo(C<Z>);

    template <int... Is, int Z>
    void waldo(B<Is...>, C<Z>);
  };

  void foo() {
    S().waldo(/*invoke completion here*/);
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):15 %s -o - | FileCheck -check-prefix=CHECK-METHOD %s
    // CHECK-METHOD-LABEL: OPENING_PAREN_LOC:
    // CHECK-METHOD-NEXT: OVERLOAD: [#void#]waldo(<#C<Z>#>)
    // CHECK-METHOD-NEXT: OVERLOAD: [#void#]waldo(<#B<>#>, C<Z>)
  }
} // namespace method
namespace function {
  template <int Z>
  void waldo(C<Z>);

  template <int... Is, int Z>
  void waldo(B<Is...>, C<Z>);

  void foo() {
    waldo(/*invoke completion here*/);
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):11 %s -o - | FileCheck -check-prefix=CHECK-FUNC %s
    // CHECK-FUNC-LABEL: OPENING_PAREN_LOC:
    // CHECK-FUNC-NEXT: OVERLOAD: [#void#]waldo(<#B<>#>, C<Z>)
    // CHECK-FUNC-NEXT: OVERLOAD: [#void#]waldo(<#C<Z>#>)
  }
} // namespace function
