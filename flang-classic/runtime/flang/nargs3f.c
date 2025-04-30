/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief getarg and iargc utility functions for Fortran programmers
 */

#include "ent3f.h"

extern int __io_get_argc();

int ENT3F(NARGS, nargs)(void)
{
  return __io_get_argc();
}
