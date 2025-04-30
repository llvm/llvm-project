/* Tests for loading and unloading of iconv modules.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 2000.

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

#include <iconv.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>


/* How many load/unload operations do we do.  */
#define TEST_ROUNDS	5000


enum state { unloaded, loaded };

struct
{
  const char *name;
  enum state state;
  iconv_t cd;
} modules[] =
{
#define MODULE(Name) { .name = #Name, .state = unloaded }
  MODULE (ISO-8859-1),
  MODULE (ISO-8859-2),
  MODULE (ISO-8859-3),
  MODULE (ISO-8859-4),
  MODULE (ISO-8859-5),
  MODULE (ISO-8859-6),
  MODULE (ISO-8859-15),
  MODULE (EUC-JP),
  MODULE (EUC-KR),
  MODULE (EUC-CN),
  MODULE (EUC-TW),
  MODULE (SJIS),
  MODULE (UHC),
  MODULE (KOI8-R),
  MODULE (BIG5),
  MODULE (BIG5HKSCS)
};
#define nmodules (sizeof (modules) / sizeof (modules[0]))


/* The test data.  */
static const char inbuf[] =
"The first step is the function to create a handle.\n"
"\n"
" - Function: iconv_t iconv_open (const char *TOCODE, const char\n"
"          *FROMCODE)\n"
"     The `iconv_open' function has to be used before starting a\n"
"     conversion.  The two parameters this function takes determine the\n"
"     source and destination character set for the conversion and if the\n"
"     implementation has the possibility to perform such a conversion the\n"
"     function returns a handle.\n"
"\n"
"     If the wanted conversion is not available the function returns\n"
"     `(iconv_t) -1'.  In this case the global variable `errno' can have\n"
"     the following values:\n"
"\n"
"    `EMFILE'\n"
"          The process already has `OPEN_MAX' file descriptors open.\n"
"\n"
"    `ENFILE'\n"
"          The system limit of open file is reached.\n"
"\n"
"    `ENOMEM'\n"
"          Not enough memory to carry out the operation.\n"
"\n"
"    `EINVAL'\n"
"          The conversion from FROMCODE to TOCODE is not supported.\n"
"\n"
"     It is not possible to use the same descriptor in different threads\n"
"     to perform independent conversions.  Within the data structures\n"
"     associated with the descriptor there is information about the\n"
"     conversion state.  This must not be messed up by using it in\n"
"     different conversions.\n"
"\n"
"     An `iconv' descriptor is like a file descriptor as for every use a\n"
"     new descriptor must be created.  The descriptor does not stand for\n"
"     all of the conversions from FROMSET to TOSET.\n"
"\n"
"     The GNU C library implementation of `iconv_open' has one\n"
"     significant extension to other implementations.  To ease the\n"
"     extension of the set of available conversions the implementation\n"
"     allows storing the necessary files with data and code in\n"
"     arbitrarily many directories.  How this extension has to be\n"
"     written will be explained below (*note glibc iconv\n"
"     Implementation::).  Here it is only important to say that all\n"
"     directories mentioned in the `GCONV_PATH' environment variable are\n"
"     considered if they contain a file `gconv-modules'.  These\n"
"     directories need not necessarily be created by the system\n"
"     administrator.  In fact, this extension is introduced to help users\n"
"     writing and using their own, new conversions.  Of course this does\n"
"     not work for security reasons in SUID binaries; in this case only\n"
"     the system directory is considered and this normally is\n"
"     `PREFIX/lib/gconv'.  The `GCONV_PATH' environment variable is\n"
"     examined exactly once at the first call of the `iconv_open'\n"
"     function.  Later modifications of the variable have no effect.\n";


static int
do_test (void)
{
  size_t count = TEST_ROUNDS;
  int result = 0;

  mtrace ();

  /* Just a seed.  */
  srandom (TEST_ROUNDS);

  while (count--)
    {
      int idx = random () % nmodules;

      if (modules[idx].state == unloaded)
	{
	  char outbuf[10000];
	  char *inptr = (char *) inbuf;
	  size_t insize = sizeof (inbuf) - 1;
	  char *outptr = outbuf;
	  size_t outsize = sizeof (outbuf);

	  /* Load the module and do the conversion.  */
	  modules[idx].cd = iconv_open ("UTF-8", modules[idx].name);

	  if (modules[idx].cd == (iconv_t) -1)
	    {
	      printf ("opening of %s failed: %m\n", modules[idx].name);
	      result = 1;
	      break;
	    }

	  modules[idx].state = loaded;

	  /* Now a simple test.  */
	  if (iconv (modules[idx].cd, &inptr, &insize, &outptr, &outsize) != 0
	      || *inptr != '\0')
	    {
	      printf ("conversion with %s failed\n", modules[idx].name);
	      result = 1;
	    }
	}
      else
	{
	  /* Unload the module.  */
	  if (iconv_close (modules[idx].cd) != 0)
	    {
	      printf ("closing of %s failed: %m\n", modules[idx].name);
	      result = 1;
	      break;
	    }

	  modules[idx].state = unloaded;
	}
    }

  for (count = 0; count < nmodules; ++count)
    if (modules[count].state == loaded && iconv_close (modules[count].cd) != 0)
      {
	printf ("closing of %s failed: %m\n", modules[count].name);
	result = 1;
      }

  return result;
}

#define TIMEOUT 30
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
