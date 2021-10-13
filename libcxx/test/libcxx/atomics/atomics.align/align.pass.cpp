//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, c++03
// REQUIRES: is-lockfree-runtime-function
// ADDITIONAL_COMPILE_FLAGS: -Wno-psabi
// ... since C++20 std::__atomic_base initializes, so we get a warning about an
// ABI change for vector variants since the constructor code for that is
// different if one were to compile with architecture-specific vector
// extensions enabled.
// This however isn't ABI breaking as it was impossible for any code to trigger
// this without using libc++ internals.

// GCC currently fails because it needs -fabi-version=6 to fix mangling of
// std::atomic when used with __attribute__((vector(X))).
// XFAIL: gcc

// This fails on PowerPC, as the LLIArr2 and Padding structs do not have
// adequate alignment, despite these types returning true for the query of
// being lock-free. This is an issue that occurs when linking in the
// PowerPC GNU libatomic library into the test.
// XFAIL: target=powerpc{{.*}}le-unknown-linux-gnu

// <atomic>

// Verify that the content of atomic<T> is properly aligned if the type is
// lock-free. This can't be observed through the atomic<T> API. It is
// nonetheless required for correctness of the implementation: lock-free implies
// that ISA instructions are used, and these instructions assume "suitable
// alignment". Supported architectures all require natural alignment for
// lock-freedom (e.g. load-linked / store-conditional, or cmpxchg).

#include <atomic>
#include <cassert>

template <typename T>
struct atomic_test : public std::__atomic_base<T> {
  atomic_test() {
    if (this->is_lock_free()) {
      using AtomicImpl = decltype(this->__a_);
      assert(alignof(AtomicImpl) >= sizeof(AtomicImpl) &&
             "expected natural alignment for lock-free type");
    }
  }
};

int main(int, char**) {

// structs and unions can't be defined in the template invocation.
// Work around this with a typedef.
#define CHECK_ALIGNMENT(T)                                                     \
  do {                                                                         \
    typedef T type;                                                            \
    atomic_test<type> t;                                                       \
  } while (0)

  CHECK_ALIGNMENT(bool);
  CHECK_ALIGNMENT(char);
  CHECK_ALIGNMENT(signed char);
  CHECK_ALIGNMENT(unsigned char);
  CHECK_ALIGNMENT(char16_t);
  CHECK_ALIGNMENT(char32_t);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  CHECK_ALIGNMENT(wchar_t);
#endif
  CHECK_ALIGNMENT(short);
  CHECK_ALIGNMENT(unsigned short);
  CHECK_ALIGNMENT(int);
  CHECK_ALIGNMENT(unsigned int);
  CHECK_ALIGNMENT(long);
  CHECK_ALIGNMENT(unsigned long);
  CHECK_ALIGNMENT(long long);
  CHECK_ALIGNMENT(unsigned long long);
  CHECK_ALIGNMENT(std::nullptr_t);
  CHECK_ALIGNMENT(void *);
  CHECK_ALIGNMENT(float);
  CHECK_ALIGNMENT(double);
  CHECK_ALIGNMENT(long double);
  CHECK_ALIGNMENT(int __attribute__((vector_size(1 * sizeof(int)))));
  CHECK_ALIGNMENT(int __attribute__((vector_size(2 * sizeof(int)))));
  CHECK_ALIGNMENT(int __attribute__((vector_size(4 * sizeof(int)))));
  CHECK_ALIGNMENT(int __attribute__((vector_size(16 * sizeof(int)))));
  CHECK_ALIGNMENT(int __attribute__((vector_size(32 * sizeof(int)))));
  CHECK_ALIGNMENT(float __attribute__((vector_size(1 * sizeof(float)))));
  CHECK_ALIGNMENT(float __attribute__((vector_size(2 * sizeof(float)))));
  CHECK_ALIGNMENT(float __attribute__((vector_size(4 * sizeof(float)))));
  CHECK_ALIGNMENT(float __attribute__((vector_size(16 * sizeof(float)))));
  CHECK_ALIGNMENT(float __attribute__((vector_size(32 * sizeof(float)))));
  CHECK_ALIGNMENT(double __attribute__((vector_size(1 * sizeof(double)))));
  CHECK_ALIGNMENT(double __attribute__((vector_size(2 * sizeof(double)))));
  CHECK_ALIGNMENT(double __attribute__((vector_size(4 * sizeof(double)))));
  CHECK_ALIGNMENT(double __attribute__((vector_size(16 * sizeof(double)))));
  CHECK_ALIGNMENT(double __attribute__((vector_size(32 * sizeof(double)))));
  CHECK_ALIGNMENT(struct Empty {});
  CHECK_ALIGNMENT(struct OneInt { int i; });
  CHECK_ALIGNMENT(struct IntArr2 { int i[2]; });
  CHECK_ALIGNMENT(struct LLIArr2 { long long int i[2]; });
  CHECK_ALIGNMENT(struct LLIArr4 { long long int i[4]; });
  CHECK_ALIGNMENT(struct LLIArr8 { long long int i[8]; });
  CHECK_ALIGNMENT(struct LLIArr16 { long long int i[16]; });
  CHECK_ALIGNMENT(struct Padding { char c; /* padding */ long long int i; });
  CHECK_ALIGNMENT(union IntFloat { int i; float f; });

  return 0;
}
