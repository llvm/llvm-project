/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file listing.c
    \brief Fortran source listing file module.
*/

#include "gbldefs.h"
#include "global.h"
#include "version.h"

static int pgpos = 1; /* line position within page of next line */
static FILE *lf;      /* listing file descriptor */
static int pgno;      /* page number of next page */

#define LPP 60

static void list_ln(const char *, const char *);

void
list_init(FILE *fd)
{
  register char **p;
  char buf[80], buf1[40], buf2[20], buf3[20];
  static const char *sevtxt[4] = {"inform", "warn", "severe", "fatal"};

  lf = fd;
  pgno = 1;

  /*  WARNING:  watch for overflowing buf  */
  if (!DBGBIT(14, 3)) {

    /* ... put out switch line. */
    sprintf(
        buf, "Switches: -%s -%s -%s -%s -%s -%s",
        (flg.asmcode ? "asm" : "noasm"), (flg.dclchk ? "dclchk" : "nodclchk"),
        (flg.debug ? "debug" : "nodebug"), (flg.dlines ? "dlines" : "nodlines"),
        (flg.line ? "line" : "noline"), (flg.list ? "list" : "nolist"));
    list_line(buf);

    /*  -idir lines:  */
    for (p = flg.idir; p && *p; p++)
      list_ln("          -idir ", *p);

    sprintf(buf, "          -inform %s -opt %d -%s -%s -%s", sevtxt[flg.inform],
            flg.opt, (flg.save ? "save" : "nosave"),
            (flg.object ? "object" : "noobject"),
            (flg.onetrip ? "onetrip" : "noonetrip"));
    list_line(buf);

    buf2[0] = buf3[0] = 0;
    if (flg.depchk) {
      if (flg.depwarn)
        strcpy(buf2, "-depchk warn");
      else
        strcpy(buf2, "-depchk on");
    } else
      strcpy(buf2, "-depchk off");
    sprintf(buf, "          %s -%s %s %s", buf2,
            (flg.standard ? "standard" : "nostandard"),
            (flg.extend_source == 132 ? "-extend" : "  "),
            (flg.profile ? "-profile" : " "));
    list_line(buf);

    strcpy(buf1, "-show");
    if (flg.include && flg.xref && flg.code)
      strcat(buf1, " all");
    else {
      if (flg.include)
        strcat(buf1, " include");
      if (flg.xref)
        strcat(buf1, " xref");
      if (flg.code)
        strcat(buf1, " code");
    }
    if (!strcmp(buf1, "-show"))
      strcpy(buf1, " ");
    buf2[0] = buf3[0] = 0;
    sprintf(buf, "          -%s -%s %s %s %s",
            (flg.symbol ? "symbol" : "nosymbol"),
            (flg.ucase ? "upcase" : "noupcase"), buf1, buf2, buf3);
    list_line(buf);

    /* ... put out filename line. */
    list_ln("\nFilename: ", gbl.src_file);
  }

  list_line("");

}

void
list_line(const char *txt)
{
  list_ln("", txt);
}

static void
list_ln(const char *beg, const char *txt)
{
  if (pgpos == 1 && !DBGBIT(14, 3)) {
    if (!lf)
      return; /* in case error msg written before file
               * opened */
    fprintf(lf, "\n\n\n%s (Version %8s)          %s      page %d\n\n",
            version.lang, version.vsn, gbl.datetime, pgno);
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
list_page(void)
{

  if (lf)
    if (!(DBGBIT(14, 3) || DBGBIT(0, 32)))
      while (pgpos != 1)
        list_line("");

}
