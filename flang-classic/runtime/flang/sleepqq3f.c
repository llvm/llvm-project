/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	sleep3f.c - Implements DFPORT SLEEPQQ subprogram.  */

#if !defined(_WIN64)
#include <unistd.h>
#endif
#include "ent3f.h"

#if defined(_WIN64)

#include <windows.h>

void ENT3F(SLEEPQQ, sleepqq)(unsigned int *t)
{
  Sleep(*t); /* MS Sleep() is in terms of milliseconds */
}
#else
void ENT3F(SLEEPQQ, sleepqq)(unsigned int *t)
{
  sleep((*t) / 1000);
}

#endif
