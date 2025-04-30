/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief gbldefs.h - syminit/symutil utility definitions
 */

#ifndef INIT
#define TM_I8
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define CNULL ((char *)0)

#define MAXIDLEN 31

#define FIELD unsigned

typedef unsigned short ILM_T;

#if defined(_WIN32) || defined(HOST_WIN)
#define DCL_INT8(name) int name : 8
#define DCL_UINT8(name) FIELD name : 8
#define DCL_INT16(name) int name : 16
#define DCL_UINT16(name) unsigned name : 16
#else
#define DCL_INT8(name) char name
#define DCL_UINT8(name) FIELD name : 8
#define DCL_INT16(name) short int name
#define DCL_UINT16(name) unsigned short int name
#endif

/* define a host type which represents 'size_t' for array extents. */
#define ISZ_T BIGINT
#define UISZ_T BIGUINT
#define ISZ_PF BIGIPFSZ

typedef int LOGICAL;
#undef TRUE
#define TRUE 1
#undef FALSE
#define FALSE 0

#define BCOPY(p, q, dt, n) memcpy((char *)(p), (char *)(q), (sizeof(dt) * (n)))
#define BZERO(p, dt, n) memset((char *)(p), 0, (sizeof(dt) * (n)))
#define FREE(p) free((char *)p)

#define NEW(p, dt, n)                                       \
  if ((p = (dt *)malloc((UINT)(sizeof(dt) * (n)))) == NULL) \
    symini_interr("out of memory", 0, 4);                   \
  else

#define NEED(n, p, dt, size, newsize)                                       \
  if (n > size) {                                                           \
    if ((p = (dt *)realloc((char *)p, ((UINT)((newsize) * sizeof(dt))))) == \
        NULL)                                                               \
      symini_interr("out of memory", 0, 4);                                 \
    size = newsize;                                                         \
  } else
