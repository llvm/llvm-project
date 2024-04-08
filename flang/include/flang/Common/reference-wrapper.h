//===-- include/flang/Common/reference-wrapper.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// clang-format off
//
// Implementation of std::reference_wrapper borrowed from libcu++
// https://github.com/NVIDIA/libcudacxx/blob/f7e6cd07ed5ba826aeac0b742feafddfedc1e400/include/cuda/std/detail/libcxx/include/__functional/reference_wrapper.h#L1
// with modifications.
//
// The original source code is distributed under the Apache License v2.0
// with LLVM Exceptions.
//
// TODO: using libcu++ is the best option for CUDA, but there is a couple
// of issues:
//   * The include paths need to be set up such that all STD header files
//     are taken from libcu++.
//   * cuda:: namespace need to be forced for all std:: references.
//
// clang-format on

#ifndef FORTRAN_COMMON_REFERENCE_WRAPPER_H
#define FORTRAN_COMMON_REFERENCE_WRAPPER_H

#include "flang/Common/api-attrs.h"
#include <functional>
#include <type_traits>

#if !defined(STD_REFERENCE_WRAPPER_UNSUPPORTED) && \
    (defined(__CUDACC__) || defined(__CUDA__)) && defined(__CUDA_ARCH__)
#define STD_REFERENCE_WRAPPER_UNSUPPORTED 1
#endif

namespace Fortran::common {

template <class _Tp>
using __remove_cvref_t = std::remove_cv_t<std::remove_reference_t<_Tp>>;
template <class _Tp, class _Up>
struct __is_same_uncvref
    : std::is_same<__remove_cvref_t<_Tp>, __remove_cvref_t<_Up>> {};

#if STD_REFERENCE_WRAPPER_UNSUPPORTED
template <class _Tp> class reference_wrapper {
public:
  // types
  typedef _Tp type;

private:
  type *__f_;

  static RT_API_ATTRS void __fun(_Tp &);
  static void __fun(_Tp &&) = delete;

public:
  template <class _Up,
      class =
          std::enable_if_t<!__is_same_uncvref<_Up, reference_wrapper>::value,
              decltype(__fun(std::declval<_Up>()))>>
  constexpr RT_API_ATTRS reference_wrapper(_Up &&__u) {
    type &__f = static_cast<_Up &&>(__u);
    __f_ = std::addressof(__f);
  }

  // access
  constexpr RT_API_ATTRS operator type &() const { return *__f_; }
  constexpr RT_API_ATTRS type &get() const { return *__f_; }

  // invoke
  template <class... _ArgTypes>
  constexpr RT_API_ATTRS typename std::invoke_result_t<type &, _ArgTypes...>
  operator()(_ArgTypes &&...__args) const {
    return std::invoke(get(), std::forward<_ArgTypes>(__args)...);
  }
};

template <class _Tp> reference_wrapper(_Tp &) -> reference_wrapper<_Tp>;

template <class _Tp>
inline constexpr RT_API_ATTRS reference_wrapper<_Tp> ref(_Tp &__t) {
  return reference_wrapper<_Tp>(__t);
}

template <class _Tp>
inline constexpr RT_API_ATTRS reference_wrapper<_Tp> ref(
    reference_wrapper<_Tp> __t) {
  return __t;
}

template <class _Tp>
inline constexpr RT_API_ATTRS reference_wrapper<const _Tp> cref(
    const _Tp &__t) {
  return reference_wrapper<const _Tp>(__t);
}

template <class _Tp>
inline constexpr RT_API_ATTRS reference_wrapper<const _Tp> cref(
    reference_wrapper<_Tp> __t) {
  return __t;
}

template <class _Tp> void ref(const _Tp &&) = delete;
template <class _Tp> void cref(const _Tp &&) = delete;
#else // !STD_REFERENCE_WRAPPER_UNSUPPORTED
using std::cref;
using std::ref;
using std::reference_wrapper;
#endif // !STD_REFERENCE_WRAPPER_UNSUPPORTED

} // namespace Fortran::common

#endif // FORTRAN_COMMON_REFERENCE_WRAPPER_H
