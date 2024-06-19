// RUN: %clang_cc1 -fsyntax-only -verify %std_cxx98-14 %s

// This is a test for an egregious hack in Clang that works around
// issues with GCC's evolution. libstdc++ 4.2.x uses __is_pod as an
// identifier (to declare a struct template like the one below), while
// GCC 4.3 and newer make __is_pod a keyword. Clang treats __is_pod as
// a keyword *unless* it is introduced following the struct keyword.

template<typename T>
struct __is_pod { // expected-warning {{using the name of the builtin '__is_pod' outside of a builtin invocation is deprecated}}
  __is_pod() {} // expected-error {{expected member name or ';' after declaration specifier}}
};

__is_pod<int> ipi; // expected-warning {{using the name of the builtin '__is_pod' outside of a builtin invocation is deprecated}}

// Ditto for __is_same.
template<typename T>
struct __is_same { // expected-warning {{using the name of the builtin '__is_same' outside of a builtin invocation is deprecated}}
};

__is_same<int> isi; // expected-warning {{using the name of the builtin '__is_same' outside of a builtin invocation is deprecated}}

// Another, similar egregious hack for __is_signed, which is a type
// trait in Embarcadero's compiler but is used as an identifier in
// libstdc++.
struct test_is_signed {
  static const bool __is_signed = true; // expected-warning {{using the name of the builtin '__is_signed' outside of a builtin invocation is deprecated}}
};

bool check_signed = test_is_signed::__is_signed; // expected-warning {{using the name of the builtin '__is_signed' outside of a builtin invocation is deprecated}}

template<bool B> struct must_be_true {};
template<> struct must_be_true<false>;

void foo() {
  bool b = __is_pod(int);
  must_be_true<__is_pod(int)> mbt;
}

// expected-warning@+1 {{declaration does not declare anything}}
struct // expected-error {{declaration of anonymous struct must be a definition}}
#pragma pack(pop)
    S {
};

#if !__has_feature(is_pod)
#  error __is_pod should still be available.
#endif
