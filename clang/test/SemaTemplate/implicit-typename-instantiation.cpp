// RUN: %clang_cc1 -fsyntax-only -verify=cxx17 -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx20 -std=c++20 %s
// cxx20-no-diagnostics

// When 'typename' is omitted from a dependent qualified type whose
// nested-name-specifier contains an intermediate dependent member
// (e.g. the 'Inner' / 'Node_traits' below), the implicit-typename recovery
// used to leave the intermediate member speculatively resolved against the
// template pattern. That nested-name-specifier did not survive instantiation,
// producing a spuriously dependent type that crashed CTAD and codegen.
// Omitting and writing 'typename' must produce the same type.

namespace intermediate_typedef {
template <class T> struct Iter { T *p; };
template <class T> struct Traits { typedef Iter<T> Const_iterator; };
template <class T> struct ListBase { typedef Traits<T> Node_traits; };

template <class T> struct List {
  // cxx17-warning@+1 {{missing 'typename' prior to dependent type name 'ListBase<T>::Node_traits::Const_iterator' is a C++20 extension}}
  typedef ListBase<T>::Node_traits::Const_iterator const_iterator;
  const_iterator begin() { return const_iterator(); }
};

struct S {};
template <class U> void use(U) {}

void test() {
  List<S> lst;
  Iter it = lst.begin(); // CTAD; previously crashed.
  use(it);
  static_assert(__is_same(decltype(it), Iter<S>));
}
} // namespace intermediate_typedef

namespace intermediate_tag {
// Reduced from llvm/llvm-project#174301.
template <class T> struct Tester {
  struct Inner { using T2 = int; };
  static void test();
};

template <class T>
// cxx17-warning@+1 {{missing 'typename' prior to dependent type name 'Tester<T>::Inner::T2' is a C++20 extension}}
Tester<T>::Inner::T2 getInnerT2() { return {}; }

template <class T> void Tester<T>::test() {
  auto x = getInnerT2<T>();
  static_assert(__is_same(decltype(x), int));
}

template struct Tester<int>;
} // namespace intermediate_tag
