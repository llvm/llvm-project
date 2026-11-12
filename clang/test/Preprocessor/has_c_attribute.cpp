// RUN: %clang_cc1 -std=c++11 %s -verify

#if __has_c_attribute(fallthrough) // expected-error {{function-like macro '__has_c_attribute' is not defined}}
#endif

#if __has_c_attribute(gnu::transparent_union) // expected-error {{function-like macro '__has_c_attribute' is not defined}}
#endif

