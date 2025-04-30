/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \brief Fortran source listing file module.
 */

#include "listing.h"
#include "gbldefs.h"
#include "global.h"
#include "version.h"

static int pgpos = 1; /* line position within page of next line */
static FILE *lf;      /* listing file descriptor */
static int pgno;      /* page number of next page */

const int LPP = 60;

static void
list_ln(const char *beg, const char *txt)
{
  if (pgpos == 1 && !DBGBIT(14, 3)) {
    if (!lf)
      return; /* in case error msg written before file
               * opened */
    fprintf(lf, "\n\n\n%s(Version %8s)          %s      page %d\n\n",
            version.product, version.vsn, gbl.datetime, pgno);
    pgno++;
    pgpos = 6;
  }

  fprintf(lf, "%s%s\n", beg, txt);
#if DEBUG
  if (DBGBIT(0, 4))
    fprintf(gbl.dbgfil, "%s%s\n", beg, txt);
#endif
  pgpos++;

  if (pgpos == LPP + 4 && !DBGBIT(14, 3)) {
    fprintf(lf, "\n\n\n");
    pgpos = 1;
  }
}

void
list_init(FILE *fd)
{
  lf = fd;
  pgno = 1;

  /*  WARNING:  watch for overflowing buf  */
  if (!DBGBIT(14, 3)) {
    /* ... put out filename line. */
    list_ln("\nFilename: ", gbl.src_file);
  }

  list_line("");
}

/*******************************************************************/

void
list_line(const char *txt)
{
  list_ln("", txt);
}

/*******************************************************************/

void
list_page(void)
{
  if (lf)
    if (!(DBGBIT(14, 3) || DBGBIT(0, 32)))
      while (pgpos != 1)
        list_line("");
}
