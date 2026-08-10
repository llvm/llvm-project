// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify

// This is a test for a hack in Clang that works around an issue with libc++'s
// <valarray> implementation. The <valarray> header contains explicit
// instantiations of functions that it declared with the internal_linkage
// attribute, which are ill-formed by [temp.explicit]p13 (and meaningless).

#ifdef BE_THE_HEADER

#pragma GCC system_header
namespace std {
  using size_t = __SIZE_TYPE__;
  template<typename T> struct valarray {
    __attribute__((internal_linkage)) valarray(size_t) {}
    __attribute__((internal_linkage)) ~valarray() {}
  };

  extern template valarray<size_t>::valarray(size_t);
  extern template valarray<size_t>::~valarray();
}

#else

#define BE_THE_HEADER
#include "libcxx_valarray_hack.cpp"

template<typename T> struct foo {
  __attribute__((internal_linkage)) void x() {};
};
extern template void foo<int>::x(); // expected-error {{explicit instantiation declaration of 'x' with internal linkage}}

#endif
