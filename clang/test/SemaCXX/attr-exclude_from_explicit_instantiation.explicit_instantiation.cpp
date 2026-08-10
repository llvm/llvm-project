// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test that explicit instantiations do not instantiate entities
// marked with the exclude_from_explicit_instantiation attribute.

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct Foo {
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION inline void non_static_member_function1();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void non_static_member_function2();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static inline void static_member_function1();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static void static_member_function2();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static int static_data_member;

  struct EXCLUDE_FROM_EXPLICIT_INSTANTIATION member_class1 {
    static void non_static_member_function() { using Fail = typename T::fail; }
  };

  struct member_class2 {
    EXCLUDE_FROM_EXPLICIT_INSTANTIATION static void non_static_member_function() { using Fail = typename T::fail; }
  };
};

template <class T>
inline void Foo<T>::non_static_member_function1() { using Fail = typename T::fail; }

template <class T>
void Foo<T>::non_static_member_function2() { using Fail = typename T::fail; }

template <class T>
inline void Foo<T>::static_member_function1() { using Fail = typename T::fail; }

template <class T>
void Foo<T>::static_member_function2() { using Fail = typename T::fail; }

template <class T>
int Foo<T>::static_data_member = T::fail;

// expected-no-diagnostics
template struct Foo<int>;
