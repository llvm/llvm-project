// RUN: %clang_cc1 -fsyntax-only -verify %s -DNO_TADD -std=c++23 -fexperimental-cxx-type-aware-allocators

namespace std {
  template <class T> struct type_identity {
    typedef T type;
  };
  enum class align_val_t : __SIZE_TYPE__ {};
  struct destroying_delete_t { explicit destroying_delete_t() = default; };
}

using size_t = __SIZE_TYPE__;


template <class Tp> struct is_const {
  static const bool value = false;
};
template <class Tp> struct is_const<Tp const> {
  static const bool value = true;
};

template <class Tp> struct is_volatile {
  static const bool value = false;
};
template <class Tp> struct is_volatile<Tp volatile> {
  static const bool value = true;
};

template <class T> static const bool is_const_v = is_const<T>::value;
template <class T> static const bool is_volatile_v = is_volatile<T>::value;

template <bool expectConst, bool expectVolatile>
struct VerifyQualifiers {
  template <typename T> void *operator new(std::type_identity<T>, size_t) throw() {
    static_assert(is_const_v<T> == expectConst); // #1
    static_assert(is_volatile_v<T> == expectVolatile); // #2
    return 0;
  }
  template <typename T> void operator delete(std::type_identity<T>, void*) {
    static_assert(is_const_v<T> == expectConst); // #3
    static_assert(is_volatile_v<T> == expectVolatile); // #4
  }
};

template <bool expectConst, bool expectVolatile> void *operator new(std::type_identity<VerifyQualifiers<expectConst, expectVolatile> > type, size_t) throw() {
  static_assert(is_const_v<typename decltype(type)::type> == expectConst); // #5
  static_assert(is_volatile_v<typename decltype(type)::type> == expectVolatile); // #6
  return 0;
}

template <bool expectConst, bool expectVolatile> void operator delete(std::type_identity<VerifyQualifiers<expectConst, expectVolatile> > type, void*) {
  static_assert(is_const_v<typename decltype(type)::type> == expectConst); // #7
  static_assert(is_volatile_v<typename decltype(type)::type> == expectVolatile); // #8
}

// Success tests
void test_member_allocators() {
  auto *unqualified_obj = new VerifyQualifiers<false, false>();
  delete unqualified_obj;
  auto *const_obj = new const VerifyQualifiers<true, false>();
  delete const_obj;
  auto *volatile_obj = new volatile VerifyQualifiers<false, true>();
  delete volatile_obj;
  auto *const_volatile_obj = new const volatile VerifyQualifiers<true, true>();
  delete const_volatile_obj;
}

void test_global_allocators() {
  auto *unqualified_obj = ::new VerifyQualifiers<false, false>();
  ::delete unqualified_obj;
  auto *const_obj = ::new const VerifyQualifiers<true, false>();
  ::delete const_obj;
  auto *volatile_obj = ::new volatile VerifyQualifiers<false, true>();
  ::delete volatile_obj;
  auto *const_volatile_obj = ::new const volatile VerifyQualifiers<true, true>();
  ::delete const_volatile_obj;
}

// Verify mismatches
void test_incorrect_member_allocators() {
  VerifyQualifiers<true, false> *incorrect_const_obj = new VerifyQualifiers<true, false>();
  // expected-error@#1 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, false>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'VerifyQualifiers<true, false>::operator new<VerifyQualifiers<true, false>>' requested here}}
  delete incorrect_const_obj;
  // expected-error@#3 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, false>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'VerifyQualifiers<true, false>::operator delete<VerifyQualifiers<true, false>>' requested here}}

  VerifyQualifiers<false, true> *incorrect_volatile_obj = new VerifyQualifiers<false, true>();
  // expected-error@#2 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<false, true>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'VerifyQualifiers<false, true>::operator new<VerifyQualifiers<false, true>>' requested here}}
  delete incorrect_volatile_obj;
  // expected-error@#4 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<false, true>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'VerifyQualifiers<false, true>::operator delete<VerifyQualifiers<false, true>>' requested here}}

  VerifyQualifiers<true, true> *incorrect_const_volatile_obj = new VerifyQualifiers<true, true>();
  // expected-error@#1 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, true>> == true'}}
  // expected-error@#2 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<true, true>> == true'}}
  // expected-note@-3 {{in instantiation of function template specialization 'VerifyQualifiers<true, true>::operator new<VerifyQualifiers<true, true>>' requested here}}
  delete incorrect_const_volatile_obj;
  // expected-error@#3 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, true>> == true'}}
  // expected-error@#4 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<true, true>> == true'}}
  // expected-note@-3 {{in instantiation of function template specialization 'VerifyQualifiers<true, true>::operator delete<VerifyQualifiers<true, true>>' requested here}}
}


// Verify mismatches
void test_incorrect_global_allocators() {
  VerifyQualifiers<true, false> *incorrect_const_obj = ::new VerifyQualifiers<true, false>();
  // expected-error@#5 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, false>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'operator new<true, false>' requested here}}
  ::delete incorrect_const_obj;
  // expected-error@#7 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, false>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'operator delete<true, false>' requested here}}

  VerifyQualifiers<false, true> *incorrect_volatile_obj = ::new VerifyQualifiers<false, true>();
  // expected-error@#6 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<false, true>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'operator new<false, true>' requested here}}
  ::delete incorrect_volatile_obj;
  // expected-error@#8 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<false, true>> == true'}}
  // expected-note@-2 {{in instantiation of function template specialization 'operator delete<false, true>' requested here}}

  VerifyQualifiers<true, true> *incorrect_const_volatile_obj = ::new VerifyQualifiers<true, true>();
  // expected-error@#5 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, true>> == true'}}
  // expected-error@#6 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<true, true>> == true'}}
  // expected-note@-3 {{in instantiation of function template specialization 'operator new<true, true>' requested here}}
  ::delete incorrect_const_volatile_obj;
  // expected-error@#7 {{static assertion failed due to requirement 'is_const_v<VerifyQualifiers<true, true>> == true'}}
  // expected-error@#8 {{static assertion failed due to requirement 'is_volatile_v<VerifyQualifiers<true, true>> == true'}}
  // expected-note@-3 {{in instantiation of function template specialization 'operator delete<true, true>' requested here}}
}
