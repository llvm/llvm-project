//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

//  template<class T, class U> complex<__promote_t<T, U>> pow(const complex<T>&, const U&);
//  template<class T, class U> complex<__promote_t<T, U>> pow(const complex<T>&, const complex<U>&);
//  template<class T, class U> complex<__promote_t<T, U>> pow(const T&, const complex<U>&);

// Test that these additional overloads are free from catching std::complex<non-floating-point>,
// which is expected by several 3rd party libraries, see https://github.com/llvm/llvm-project/issues/109858.
//
// Note that we reserve the right to break this in the future if we have a reason to, but for the time being,
// make sure we don't break this property unintentionally.
#include <cassert>
#include <cmath>
#include <complex>
#include <type_traits>

#include "test_macros.h"

namespace usr {
struct usr_tag {};

template <class T, class U>
typename std::enable_if<(std::is_same<T, usr_tag>::value && std::is_floating_point<U>::value) ||
                            (std::is_floating_point<T>::value && std::is_same<U, usr_tag>::value),
                        int>::type
pow(const T&, const std::complex<U>&) {
  return std::is_same<T, usr_tag>::value ? 0 : 1;
}

template <class T, class U>
typename std::enable_if<(std::is_same<T, usr_tag>::value && std::is_floating_point<U>::value) ||
                            (std::is_floating_point<T>::value && std::is_same<U, usr_tag>::value),
                        int>::type
pow(const std::complex<T>&, const U&) {
  return std::is_same<U, usr_tag>::value ? 2 : 3;
}

template <class T, class U>
typename std::enable_if<(std::is_same<T, usr_tag>::value && std::is_floating_point<U>::value) ||
                            (std::is_floating_point<T>::value && std::is_same<U, usr_tag>::value),
                        int>::type
pow(const std::complex<T>&, const std::complex<U>&) {
  return std::is_same<T, usr_tag>::value ? 4 : 5;
}
} // namespace usr

int main(int, char**) {
  using std::pow;
  using usr::pow;

  usr::usr_tag tag;
  const std::complex<usr::usr_tag> ctag;

  assert(pow(tag, std::complex<float>(1.0f)) == 0);
  assert(pow(std::complex<float>(1.0f), tag) == 2);
  assert(pow(tag, std::complex<double>(1.0)) == 0);
  assert(pow(std::complex<double>(1.0), tag) == 2);
  assert(pow(tag, std::complex<long double>(1.0l)) == 0);
  assert(pow(std::complex<long double>(1.0l), tag) == 2);

  assert(pow(1.0f, ctag) == 1);
  assert(pow(ctag, 1.0f) == 3);
  assert(pow(1.0, ctag) == 1);
  assert(pow(ctag, 1.0) == 3);
  assert(pow(1.0l, ctag) == 1);
  assert(pow(ctag, 1.0l) == 3);

  assert(pow(ctag, std::complex<float>(1.0f)) == 4);
  assert(pow(std::complex<float>(1.0f), ctag) == 5);
  assert(pow(ctag, std::complex<double>(1.0)) == 4);
  assert(pow(std::complex<double>(1.0), ctag) == 5);
  assert(pow(ctag, std::complex<long double>(1.0l)) == 4);
  assert(pow(std::complex<long double>(1.0l), ctag) == 5);

  return 0;
}
