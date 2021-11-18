//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "vector"

_LIBCPP_BEGIN_NAMESPACE_STD

void __vector_base_common<true>::__throw_length_error() const {
    _VSTD::__throw_length_error("vector");
}

void __vector_base_common<true>::__throw_out_of_range() const {
    _VSTD::__throw_out_of_range("vector");
}

_LIBCPP_END_NAMESPACE_STD
