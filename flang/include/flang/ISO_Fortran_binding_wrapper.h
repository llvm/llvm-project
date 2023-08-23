/*===-- include/flang/ISO_Fortran_binding_wrapper.h ---------------*- C++ -*-===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * ===-----------------------------------------------------------------------===
 */

#ifndef FORTRAN_ISO_FORTRAN_BINDING_WRAPPER_H_
#define FORTRAN_ISO_FORTRAN_BINDING_WRAPPER_H_

/* A thin wrapper around flang/include/ISO_Fortran_binding.h
 * This header file must be included when ISO_Fortran_binding.h
 * definitions/declarations are needed in Flang compiler/runtime
 * sources. The inclusion of Runtime/api-attrs.h below sets up
 * proper values for the macros used in ISO_Fortran_binding.h
 * for the device offload builds.
 * flang/include/ISO_Fortran_binding.h is made a standalone
 * header file so that it can be used on its own in users'
 * C/C++ programs.
 */

/* clang-format off */
#include "Runtime/api-attrs.h"
#include "ISO_Fortran_binding.h"
/* clang-format on */

#endif /* FORTRAN_ISO_FORTRAN_BINDING_WRAPPER_H_ */
