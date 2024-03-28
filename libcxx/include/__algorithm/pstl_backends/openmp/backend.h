//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_BACKEND_H

#include <__config>

#include <__algorithm/pstl_backends/openmp/omp_offload.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17

#  if defined(_LIBCPP_PSTL_BACKEND_OPENMP)
#    if !defined(_OPENMP)
#      error "PSTL is configured to use the OpenMP backend, but OpenMP is not enabled. Did you compile with -fopenmp?"
#    elif (defined(_OPENMP) && _OPENMP < 201511)
#      error                                                                                                           \
          "OpenMP target offloading has been supported since OpenMP version 4.5 (201511). Please use a more recent version of OpenMP."
#    endif
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

struct __omp_backend_tag {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKENDS_OPENMP_BACKEND_BACKEND_H
