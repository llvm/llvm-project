// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wno-nullability-declspec %s -verify -Wnullable-to-nonnull-conversion -I%S/Inputs

class Foo;
using Foo1 = Foo _Nonnull; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'Foo'}}
class _Nullable Foo;
using Foo2 = Foo _Nonnull;
class Foo;
using Foo3 = Foo _Nonnull;

template <class T>
class Bar;
using Bar1 = Bar<int> _Nonnull; // expected-error{{nullability specifier '_Nonnull' cannot be applied to non-pointer type 'Bar<int>'}}
template <class T>
class _Nullable Bar;
using Bar2 = Bar<int> _Nonnull;
template <class T>
class Bar;
using Bar3 = Bar<int> _Nonnull;

namespace std {
  template<class T> class unique_ptr;
  using UP1 = unique_ptr<int> _Nonnull;
  class X { template<class T> friend class unique_ptr; };
  using UP2 = unique_ptr<int> _Nonnull;
  template<class T> class unique_ptr;
  using UP3 = unique_ptr<int> _Nonnull;
}
