/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int
main (int argc, char **argv)
{
  time_t t;
  struct tm *tp;
  struct tm tbuf;
  int lose = 0;

  --argc;
  ++argv;

  do
    {
      char buf[BUFSIZ];
      if (argc > 0)
	{
	  static char buf[BUFSIZ];
	  sprintf(buf, "TZ=%s", *argv);
	  if (putenv(buf))
	    {
	      puts("putenv failed.");
	      lose = 1;
	    }
	  else
	    puts (buf);
	}
      tzset();
      tbuf.tm_year = 72;
      tbuf.tm_mon = 0;
      tbuf.tm_mday = 31;
      tbuf.tm_hour = 6;
      tbuf.tm_min = 14;
      tbuf.tm_sec = 50;
      tbuf.tm_isdst = -1;
    doit:;
      t = mktime(&tbuf);
      if (t == (time_t) -1)
	{
	  puts("mktime() failed?");
	  lose = 1;
	}
      tp = localtime(&t);
      if (tp == NULL)
	{
	  puts("localtime() failed.");
	  lose = 1;
	}
      else if (strftime (buf, sizeof (buf), "%a %b %d %X %Z %Y", tp) == 0)
	{
	  puts("strftime() failed.");
	  lose = 1;
	}
      else
	puts(buf);
      if (tbuf.tm_year == 101)
	{
	  tbuf.tm_year = 97;
	  tbuf.tm_mon = 0;
	  goto doit;
	}
      ++argv;
    } while (--argc > 0);

  {
#define	SIZE	256
    char buffer[SIZE];
    time_t curtime;
    struct tm *loctime;

    curtime = time (NULL);

    loctime = localtime (&curtime);

    fputs (asctime (loctime), stdout);

    strftime (buffer, SIZE, "Today is %A, %B %d.\n", loctime);
    fputs (buffer, stdout);
    strftime (buffer, SIZE, "The time is %I:%M %p.\n", loctime);
    fputs (buffer, stdout);

    loctime->tm_year = 72;
    loctime->tm_mon = 8;
    loctime->tm_mday = 12;
    loctime->tm_hour = 20;
    loctime->tm_min = 49;
    loctime->tm_sec = 05;
    curtime = mktime (loctime);
    strftime (buffer, SIZE, "%D %T was %w the %jth.\n", loctime);
    fputs (buffer, stdout);
  }

  return (lose ? EXIT_FAILURE : EXIT_SUCCESS);
}
