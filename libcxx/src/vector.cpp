//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_AVAILABILITY_MINIMUM_HEADER_VERSION < 15

template <bool>
struct __vector_base_common;

template <>
struct __vector_base_common<true> {
  [[noreturn]] _LIBCPP_EXPORTED_FROM_ABI void __throw_length_error() const;
  [[noreturn]] _LIBCPP_EXPORTED_FROM_ABI void __throw_out_of_range() const;
};

void __vector_base_common<true>::__throw_length_error() const { std::__throw_length_error("vector"); }

void __vector_base_common<true>::__throw_out_of_range() const { std::__throw_out_of_range("vector"); }

#endif // _LIBCPP_AVAILABILITY_MINIMUM_HEADER_VERSION < 15

_LIBCPP_END_NAMESPACE_STD
