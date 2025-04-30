/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if DEBUG
#include <stdarg.h>

/* print a message, continue */
#define Trace(a) TraceOutput a

static void
TraceOutput(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);

  if (DBGBIT(TRACEFLAG, TRACEBIT)) {
    if (gbl.dbgfil) {
      fprintf(gbl.dbgfil, TRACESTRING);
      vfprintf(gbl.dbgfil, fmt, argptr);
      fprintf(gbl.dbgfil, "\n");
    } else {
      fprintf(stderr, TRACESTRING);
      fprintf(stderr, "Trace: ");
      vfprintf(stderr, fmt, argptr);
      fprintf(stderr, "\n");
    }
    va_end(argptr);
  }
} /* TraceOutput */
#else
/* DEBUG not set */
/* eliminate the trace output */
#define Trace(a)
#endif
