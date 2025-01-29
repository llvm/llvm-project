//===-- include/flang/Runtime/complex.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A single way to expose C++ complex class in files that can be used
// in F18 runtime build. With inclusion of this file std::complex
// and the related names become available, though, they may correspond
// to alternative definitions (e.g. from cuda::std namespace).

#ifndef FORTRAN_RUNTIME_COMPLEX_H
#define FORTRAN_RUNTIME_COMPLEX_H

#include "flang/Common/api-attrs.h"

#if RT_USE_LIBCUDACXX && defined(RT_DEVICE_COMPILATION)
#include <cuda/std/complex>
namespace Fortran::runtime::rtcmplx {
using cuda::std::complex;
using cuda::std::conj;
} // namespace Fortran::runtime::rtcmplx
#else // !(RT_USE_LIBCUDACXX && defined(RT_DEVICE_COMPILATION))
#include <complex>
namespace Fortran::runtime::rtcmplx {
using std::complex;
using std::conj;
} // namespace Fortran::runtime::rtcmplx
#endif // !(RT_USE_LIBCUDACXX && defined(RT_DEVICE_COMPILATION))

#endif // FORTRAN_RUNTIME_COMPLEX_H
