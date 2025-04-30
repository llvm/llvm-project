/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ASPRINTF_H
#define ASPRINTF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

/**
 * Sets `char **' pointer to be a buffer
 * large enough to hold the formatted string
 * accepting a `va_list' args of variadic
 * arguments.
 */

int
vasprintf (char **, const char *, va_list);

/**
 * Sets `char **' pointer to be a buffer
 * large enough to hold the formatted
 * string accepting `n' arguments of
 * variadic arguments.
 */

int
asprintf (char **, const char *, ...);

#ifdef __cplusplus
}
#endif
#endif