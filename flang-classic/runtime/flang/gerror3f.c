/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	gerror3f.c - Implements LIB3F gerror subprogram.  */

#include <string.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"
#include "error.h"

#define Ftn_errmsg __fortio_errmsg

void ENT3F(GERROR, gerror)(DCHAR(str) DCLEN(str))
{
  char *p;

  p = strerror(__io_errno());
  __fcp_cstr(CADR(str), CLEN(str), p);
  return;
}

void ENT3F(GET_IOSTAT_MSG, get_iostat_msg)(int *ios, DCHAR(str) DCLEN(str))
{
  const char *p;
  p = Ftn_errmsg(*ios);
  __fcp_cstr(CADR(str), CLEN(str), p);
}

/* for -Msecond_underscore */
void ENT3F(GET_IOSTAT_MSG_, get_iostat_msg_)(int *ios, DCHAR(str) DCLEN(str))
{
  const char *p;
  p = Ftn_errmsg(*ios);
  __fcp_cstr(CADR(str), CLEN(str), p);
}
