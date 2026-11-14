// RUN: %clang_cc1 -std=c++14 -verify=expected,ref,cxx20 %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,ref,cxx20 %s
// RUN: %clang_cc1 -std=c++23 -verify=expected,ref %s
// RUN: %clang_cc1 -std=c++23 -fexperimental-new-constant-interpreter -verify=expected %s

namespace std {
struct type_info {
  virtual ~type_info();
};
}

// Ensure that diagnosing virtual dispatch on type_info object does not crash (GH237168).
constexpr void destroy_type_info() { // cxx20-error {{constexpr function never produces a constant expression}}
  typeid(int).~type_info(); // ref-note {{destruction of object 'typeid(int)' whose value is not known}}
}

#if __cplusplus > 202002L
static_assert((destroy_type_info(), true)); // expected-error {{static assertion expression is not an integral constant expression}} \
                                           // ref-note {{in call to 'destroy_type_info()'}}
#endif
