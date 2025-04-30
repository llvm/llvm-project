/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	hostnm3f.c - Implements LIB3F hostnm subprogram.  */

#if !defined(_WIN64)

#include <unistd.h>
/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"

int ENT3F(HOSTNM, hostnm)(DCHAR(nm) DCLEN(nm))
{
  char *nm = CADR(nm);
  int len = CLEN(nm);
  int i;

  i = gethostname(nm, len);
  if (i < 0)
    i = __io_errno();
  else {
    /* note: last char stored is null character; gethostname() does
     *       not return the length of the name
     */
    for (i = 0; i < len; i++)
      if (nm[i] == '\0')
        break;
    /* i is position of null character, or len if not found */
    while (i < len) {
      nm[i] = ' ';
      i++;
    }
    i = 0;
  }
  return i;
}

#endif /* !_WIN64 */
