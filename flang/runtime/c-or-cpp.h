//===-- runtime/c-or-cpp.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_C_OR_CPP_H_
#define FORTRAN_RUNTIME_C_OR_CPP_H_

#ifdef __cplusplus
#define IF_CPLUSPLUS(x) x
#define IF_NOT_CPLUSPLUS(x)
#define DEFAULT_VALUE(x) = (x)
#else
#include <stdbool.h>
#define IF_CPLUSPLUS(x)
#define IF_NOT_CPLUSPLUS(x) x
#define DEFAULT_VALUE(x)
#endif

#define FORTRAN_EXTERN_C_BEGIN IF_CPLUSPLUS(extern "C" {)
#define FORTRAN_EXTERN_C_END IF_CPLUSPLUS( \
  })
#define NORETURN IF_CPLUSPLUS([[noreturn]])
#define NO_ARGUMENTS IF_NOT_CPLUSPLUS(void)

#endif // FORTRAN_RUNTIME_C_OR_CPP_H_
