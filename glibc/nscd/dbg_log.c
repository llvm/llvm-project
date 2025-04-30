/* Copyright (c) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Thorsten Kukuk <kukuk@vt.uni-paderborn.de>, 1998.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include "dbg_log.h"
#include "nscd.h"

/* if in debug mode and we have a debug file, we write the messages to it,
   if in debug mode and no debug file, we write the messages to stderr,
   else to syslog.  */

static char *logfilename;
FILE *dbgout;
int debug_level;

void
set_logfile (const char *logfile)
{
  logfilename = strdup (logfile);
}

int
init_logfile (void)
{
  if (logfilename)
    {
      dbgout = fopen64 (logfilename, "a");
      return dbgout == NULL ? 0 : 1;
    }
  return 1;
}

void
dbg_log (const char *fmt,...)
{
  va_list ap;
  char msg2[512];

  va_start (ap, fmt);
  vsnprintf (msg2, sizeof (msg2), fmt, ap);

  if (debug_level > 0)
    {
      time_t t = time (NULL);

      struct tm now;
      localtime_r (&t, &now);

      char buf[256];
      strftime (buf, sizeof (buf), "%c", &now);

      char msg[1024];
      snprintf (msg, sizeof (msg), "%s - %d: %s%s", buf, getpid (), msg2,
		msg2[strlen (msg2) - 1] == '\n' ? "" : "\n");
      if (dbgout)
	{
	  fputs (msg, dbgout);
	  fflush (dbgout);
	}
      else
	fputs (msg, stderr);
    }
  else
    syslog (LOG_NOTICE, "%d %s", getpid (), msg2);

  va_end (ap);
}
