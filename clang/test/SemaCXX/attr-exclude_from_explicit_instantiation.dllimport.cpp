// RUN: %clang_cc1 -triple x86_64-unknown-windows -fms-extensions -verify %s

// Test that an entity marked as both dllimport and exclude_from_explicit_instantiation
// isn't instantiated.

#define DLLIMPORT __declspec(dllimport)
#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct DLLIMPORT Foo {
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void x();
};

template <class T>
struct Bar {
  DLLIMPORT EXCLUDE_FROM_EXPLICIT_INSTANTIATION inline void x();
};

template <class T>
void Foo<T>::x() { using Fail = typename T::fail; }

template <class T>
DLLIMPORT inline void Bar<T>::x() { using Fail = typename T::fail; }

// expected-no-diagnostics
template struct Foo<int>;
template struct Bar<int>;
