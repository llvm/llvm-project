//===-- include/flang/Common/variant.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A single way to expose C++ variant class in files that can be used
// in F18 runtime build. With inclusion of this file std::variant
// and the related names become available, though, they may correspond
// to alternative definitions (e.g. from cuda::std namespace).

#ifndef FORTRAN_COMMON_VARIANT_H
#define FORTRAN_COMMON_VARIANT_H

#if RT_USE_LIBCUDACXX
#include <cuda/std/variant>
namespace std {
using cuda::std::get;
using cuda::std::monostate;
using cuda::std::variant;
using cuda::std::variant_size_v;
using cuda::std::visit;
} // namespace std
#else // !RT_USE_LIBCUDACXX
#include <variant>
#endif // !RT_USE_LIBCUDACXX

#endif // FORTRAN_COMMON_VARIANT_H
