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

// initializer_list is included to load bits/c++config, which can't be included
// directly and which defines a macro we need to redefine.
#include <initializer_list>

// Redefine problematic abort macro to avoid defining symbols related to
// std::exception. std::exception is defined in the C++ runtime library, which
// we do not want the Fortran runtime library to depend on.
#undef _GLIBCXX_THROW_OR_ABORT
#define _GLIBCXX_THROW_OR_ABORT(_EXC) (__builtin_abort())

#include <variant>
#endif // !RT_USE_LIBCUDACXX

#endif // FORTRAN_COMMON_VARIANT_H
