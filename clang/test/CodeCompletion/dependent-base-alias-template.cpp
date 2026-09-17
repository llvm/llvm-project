template <typename T> struct Base {
  void fromBase();
};

// A dependent base class can be named through an alias template. libstdc++'s
// std::allocator does exactly this: it derives from __allocator_base<T>, which
// is an alias template for __new_allocator<T>.
template <typename T> using AliasBase = Base<T>;
template <typename T> using AliasOfAlias = AliasBase<T>;

template <typename T> struct Derived : AliasBase<T> {
  void ownMember();
};

template <typename T> struct DerivedTwice : AliasOfAlias<T> {
  void ownMember();
};

template <typename T> void f(Derived<T> d) {
  d.
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):5 -std=c++17 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: COMPLETION: Base (InBase) : Base::
// CHECK-CC1: COMPLETION: Derived : Derived::
// CHECK-CC1: COMPLETION: fromBase (InBase) : [#void#][#Base<T>::#]fromBase()
// CHECK-CC1: COMPLETION: ownMember : [#void#]ownMember()

template <typename T> void g(DerivedTwice<T> d) {
  d.
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):5 -std=c++17 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: COMPLETION: Base (InBase) : Base::
// CHECK-CC2: COMPLETION: DerivedTwice : DerivedTwice::
// CHECK-CC2: COMPLETION: fromBase (InBase) : [#void#][#Base<T>::#]fromBase()
// CHECK-CC2: COMPLETION: ownMember : [#void#]ownMember()
