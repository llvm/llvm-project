// RUN: %clang_cc1 -Wno-unused-local-typedef -fsyntax-only -verify %s

// Test that extern instantiation declarations cause members marked with
// the exclude_from_explicit_instantiation attribute to be instantiated in
// the current TU.

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))

template <class T>
struct Foo {
  EXCLUDE_FROM_EXPLICIT_INSTANTIATION inline void non_static_member_function1();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION void non_static_member_function2();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static inline void static_member_function1();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static void static_member_function2();

  EXCLUDE_FROM_EXPLICIT_INSTANTIATION static int static_data_member;

  struct EXCLUDE_FROM_EXPLICIT_INSTANTIATION member_class1 {
    static void static_member_function() {
      using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
    }
  };

  struct member_class2 {
    EXCLUDE_FROM_EXPLICIT_INSTANTIATION static void static_member_function() {
      using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
    }
  };
};

template <class T>
inline void Foo<T>::non_static_member_function1() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

template <class T>
void Foo<T>::non_static_member_function2() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

template <class T>
inline void Foo<T>::static_member_function1() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

template <class T>
void Foo<T>::static_member_function2() {
  using Fail = typename T::invalid; // expected-error{{no type named 'invalid' in 'Empty'}}
}

template <class T>
int Foo<T>::static_data_member = T::invalid; // expected-error{{no member named 'invalid' in 'Empty'}}

struct Empty { };
extern template struct Foo<Empty>;

int main() {
  Foo<Empty> foo;
  foo.non_static_member_function1();                   // expected-note{{in instantiation of}}
  foo.non_static_member_function2();                   // expected-note{{in instantiation of}}
  Foo<Empty>::static_member_function1();               // expected-note{{in instantiation of}}
  Foo<Empty>::static_member_function2();               // expected-note{{in instantiation of}}
  (void)foo.static_data_member;                        // expected-note{{in instantiation of}}
  Foo<Empty>::member_class1::static_member_function(); // expected-note{{in instantiation of}}
  Foo<Empty>::member_class2::static_member_function(); // expected-note{{in instantiation of}}
}
