// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

#ifndef __GXX_EXPERIMENTAL_CXX0X__
#define __CONCAT(__X, __Y) __CONCAT1(__X, __Y)
#define __CONCAT1(__X, __Y) __X ## __Y

#define static_assert(__b, __m) \
  typedef int __CONCAT(__sa, __LINE__)[__b ? 1 : -1]
#endif

template <int N> class IntArray {
  int elems[N];
};

static_assert(sizeof(IntArray<10>) == sizeof(int) * 10, "Array size mismatch");
static_assert(sizeof(IntArray<1>) == sizeof(int) * 1, "Array size mismatch");

template <typename T> class TenElementArray {
  int elems[10];
};

static_assert(sizeof(TenElementArray<int>) == sizeof(int) * 10, "Array size mismatch");

template<typename T, int N> class Array {
  T elems[N];
};

static_assert(sizeof(Array<int, 10>) == sizeof(int) * 10, "Array size mismatch");
