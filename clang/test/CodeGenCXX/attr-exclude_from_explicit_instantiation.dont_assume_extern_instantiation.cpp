// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -O0 -o - %s | FileCheck %s

// Test that we do not assume that entities marked with the
// exclude_from_explicit_instantiation attribute are instantiated
// in another TU when an extern template instantiation declaration
// is present. We test that by making sure that definitions are
// generated in this TU despite there being an extern template
// instantiation declaration, which is normally not the case.

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct Foo {
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION        inline void non_static_member_function1();
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION               void non_static_member_function2();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static inline void static_member_function1();
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static        void static_member_function2();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static        int static_data_member;

  struct EXCLUDE_FROM_EXPLICIT_INSTANTIATION member_class1 {
    static void static_member_function() { }
  };

  struct member_class2 {
    EXCLUDE_FROM_EXPLICIT_INSTANTIATION static void static_member_function() { }
  };
};

template <class T> inline void Foo<T>::non_static_member_function1() { }
template <class T>        void Foo<T>::non_static_member_function2() { }

template <class T> inline void Foo<T>::static_member_function1() { }
template <class T>        void Foo<T>::static_member_function2() { }

template <class T>        int Foo<T>::static_data_member = 0;

extern template struct Foo<int>;

void use() {
  Foo<int> f;

  // An inline non-static member function marked with the attribute is not
  // part of the extern template declaration, so a definition must be emitted
  // in this TU.
  // CHECK-DAG: define linkonce_odr void @_ZN3FooIiE27non_static_member_function1Ev
  f.non_static_member_function1();

  // A non-inline non-static member function marked with the attribute is
  // not part of the extern template declaration, so a definition must be
  // emitted in this TU.
  // CHECK-DAG: define linkonce_odr void @_ZN3FooIiE27non_static_member_function2Ev
  f.non_static_member_function2();

  // An inline static member function marked with the attribute is not
  // part of the extern template declaration, so a definition must be
  // emitted in this TU.
  // CHECK-DAG: define linkonce_odr void @_ZN3FooIiE23static_member_function1Ev
  Foo<int>::static_member_function1();

  // A non-inline static member function marked with the attribute is not
  // part of the extern template declaration, so a definition must be
  // emitted in this TU.
  // CHECK-DAG: define linkonce_odr void @_ZN3FooIiE23static_member_function2Ev
  Foo<int>::static_member_function2();

  // A static data member marked with the attribute is not part of the
  // extern template declaration, so a definition must be emitted in this TU.
  // CHECK-DAG: @_ZN3FooIiE18static_data_memberE = linkonce_odr global
  int& odr_use = Foo<int>::static_data_member;

  // A member class marked with the attribute is not part of the extern
  // template declaration (it is not recursively instantiated), so its member
  // functions must be emitted in this TU.
  // CHECK-DAG: define linkonce_odr void @_ZN3FooIiE13member_class122static_member_functionEv
  Foo<int>::member_class1::static_member_function();

  // A member function marked with the attribute in a member class is not
  // part of the extern template declaration of the parent class template, so
  // it must be emitted in this TU.
  // CHECK-DAG: define linkonce_odr void @_ZN3FooIiE13member_class222static_member_functionEv
  Foo<int>::member_class2::static_member_function();
}
