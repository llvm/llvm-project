/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief External declarations for libpgc routines defined in mpalloc.c.
 */

#include <stdlib.h>

extern void * _mp_malloc(size_t n);
extern void * _mp_calloc(size_t n, size_t);
extern void * _mp_realloc(void *p, size_t n);
extern void * _mp_realloc(void *p, size_t n);
extern void _mp_free(void *p);
