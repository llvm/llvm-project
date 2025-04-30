/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 2000.

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

#include <ctype.h>
#include <locale.h>
#include <langinfo.h>
#include <stdio.h>
#include <string.h>


static const char lower[] = "abcdefghijklmnopqrstuvwxyz";
static const char upper[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char digits[] = "0123456789";
static const char cntrl[] = "\
\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\
\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f ";


static struct classes
{
  const char *name;
  int mask;
} classes[] =
{
#define ENTRY(name) { #name, _IS##name }
  ENTRY (upper),
  ENTRY (lower),
  ENTRY (alpha),
  ENTRY (digit),
  ENTRY (xdigit),
  ENTRY (space),
  ENTRY (print),
  ENTRY (graph),
  ENTRY (blank),
  ENTRY (cntrl),
  ENTRY (punct),
  ENTRY (alnum)
};
#define nclasses (sizeof (classes) / sizeof (classes[0]))


#define FAIL(str, args...) \
  {									      \
    printf ("      " str "\n", ##args);					      \
    ++errors;								      \
  }


static int
do_test (void)
{
  const char *cp;
  const char *cp2;
  int errors = 0;
  char *inpline = NULL;
  size_t inplinelen = 0;
  char *resline = NULL;
  size_t reslinelen = 0;
  size_t n;
  const unsigned short int *__ctype_b;

  setlocale (LC_ALL, "");

  printf ("Testing the ctype data of the `%s' locale\n",
	  setlocale (LC_CTYPE, NULL));

  __ctype_b = ((const unsigned short *) nl_langinfo (_NL_CTYPE_CLASS)) + 128;

#if 0
  /* Just for debugging.  */

  /* Contents of the class array.  */
  printf ("\
upper = %04x  lower = %04x  alpha = %04x  digit = %04x  xdigit = %04x\n\
space = %04x  print = %04x  graph = %04x  blank = %04x  cntrl  = %04x\n\
punct = %04x  alnum = %04x\n",
	  _ISupper, _ISlower, _ISalpha, _ISdigit, _ISxdigit,
	  _ISspace, _ISprint, _ISgraph, _ISblank, _IScntrl,
	  _ISpunct, _ISalnum);

  while (n < 256)
    {
      if (n % 8 == 0)
	printf ("%02x: ", n);
      printf ("%04x%s", __ctype_b[n], (n + 1) % 8 == 0 ? "\n" : " ");
      ++n;
    }
#endif

  puts ("  Test of ASCII character range\n    special NUL byte handling");
  if (isupper ('\0'))
    FAIL ("isupper ('\\0') is true");
  if (islower ('\0'))
    FAIL ("islower ('\\0') is true");
  if (isalpha ('\0'))
    FAIL ("isalpha ('\\0') is true");
  if (isdigit ('\0'))
    FAIL ("isdigit ('\\0') is true");
  if (isxdigit ('\0'))
    FAIL ("isxdigit ('\\0') is true");
  if (isspace ('\0'))
    FAIL ("isspace ('\\0') is true");
  if (isprint ('\0'))
    FAIL ("isprint ('\\0') is true");
  if (isgraph ('\0'))
    FAIL ("isgraph ('\\0') is true");
  if (isblank ('\0'))
    FAIL ("isblank ('\\0') is true");
  if (! iscntrl ('\0'))
    FAIL ("iscntrl ('\\0') not true");
  if (ispunct ('\0'))
    FAIL ("ispunct ('\\0') is true");
  if (isalnum ('\0'))
    FAIL ("isalnum ('\\0') is true");

  puts ("    islower()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (! islower (*cp))
      FAIL ("islower ('%c') not true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (islower (*cp))
      FAIL ("islower ('%c') is true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (islower (*cp))
      FAIL ("islower ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (islower (*cp))
      FAIL ("islower ('\\x%02x') is true", *cp);

  puts ("    isupper()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (isupper (*cp))
      FAIL ("isupper ('%c') is true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (! isupper (*cp))
      FAIL ("isupper ('%c') not true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (isupper (*cp))
      FAIL ("isupper ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (isupper (*cp))
      FAIL ("isupper ('\\x%02x') is true", *cp);

  puts ("    isalpha()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (! isalpha (*cp))
      FAIL ("isalpha ('%c') not true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (! isalpha (*cp))
      FAIL ("isalpha ('%c') not true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (isalpha (*cp))
      FAIL ("isalpha ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (isalpha (*cp))
      FAIL ("isalpha ('\\x%02x') is true", *cp);

  puts ("    isdigit()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (isdigit (*cp))
      FAIL ("isdigit ('%c') is true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (isdigit (*cp))
      FAIL ("isdigit ('%c') is true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (! isdigit (*cp))
      FAIL ("isdigit ('%c') not true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (isdigit (*cp))
      FAIL ("isdigit ('\\x%02x') is true", *cp);

  puts ("    isxdigit()");
  for (cp = lower; *cp != '\0'; ++cp)
    if ((! isxdigit (*cp) && cp - lower < 6)
	|| (isxdigit (*cp) && cp - lower >= 6))
      FAIL ("isxdigit ('%c') %s true", *cp, cp - upper < 6 ? "not" : "is");
  for (cp = upper; *cp != '\0'; ++cp)
    if ((! isxdigit (*cp) && cp - upper < 6)
	|| (isxdigit (*cp) && cp - upper >= 6))
      FAIL ("isxdigit ('%c') %s true", *cp, cp - upper < 6 ? "not" : "is");
  for (cp = digits; *cp != '\0'; ++cp)
    if (! isxdigit (*cp))
      FAIL ("isxdigit ('%c') not true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (isxdigit (*cp))
      FAIL ("isxdigit ('\\x%02x') is true", *cp);

  puts ("    isspace()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (isspace (*cp))
      FAIL ("isspace ('%c') is true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (isspace (*cp))
      FAIL ("isspace ('%c') is true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (isspace (*cp))
      FAIL ("isspace ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if ((isspace (*cp) && ((*cp < '\x09' || *cp > '\x0d') && *cp != ' '))
	|| (! isspace (*cp)
	    && ((*cp >= '\x09' && *cp <= '\x0d') || *cp == ' ')))
      FAIL ("isspace ('\\x%02x') %s true", *cp,
	    (*cp < '\x09' || *cp > '\x0d') ? "is" : "not");

  puts ("    isprint()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (! isprint (*cp))
      FAIL ("isprint ('%c') not true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (! isprint (*cp))
      FAIL ("isprint ('%c') not true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (! isprint (*cp))
      FAIL ("isprint ('%c') not true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if ((isprint (*cp) && *cp != ' ')
	|| (! isprint (*cp) && *cp == ' '))
      FAIL ("isprint ('\\x%02x') is true", *cp);

  puts ("    isgraph()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (! isgraph (*cp))
      FAIL ("isgraph ('%c') not true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (! isgraph (*cp))
      FAIL ("isgraph ('%c') not true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (! isgraph (*cp))
      FAIL ("isgraph ('%c') not true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (isgraph (*cp))
      FAIL ("isgraph ('\\x%02x') is true", *cp);

  puts ("    isblank()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (isblank (*cp))
      FAIL ("isblank ('%c') is true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (isblank (*cp))
      FAIL ("isblank ('%c') is true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (isblank (*cp))
      FAIL ("isblank ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if ((isblank (*cp) && *cp != '\x09' && *cp != ' ')
	|| (! isblank (*cp) && (*cp == '\x09' || *cp == ' ')))
      FAIL ("isblank ('\\x%02x') %s true", *cp, *cp != '\x09' ? "is" : "not");

  puts ("    iscntrl()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (iscntrl (*cp))
      FAIL ("iscntrl ('%c') is true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (iscntrl (*cp))
      FAIL ("iscntrl ('%c') is true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (iscntrl (*cp))
      FAIL ("iscntrl ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if ((iscntrl (*cp) && *cp == ' ')
	|| (! iscntrl (*cp) && *cp != ' '))
      FAIL ("iscntrl ('\\x%02x') not true", *cp);

  puts ("    ispunct()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (ispunct (*cp))
      FAIL ("ispunct ('%c') is true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (ispunct (*cp))
      FAIL ("ispunct ('%c') is true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (ispunct (*cp))
      FAIL ("ispunct ('%c') is true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (ispunct (*cp))
      FAIL ("ispunct ('\\x%02x') is true", *cp);

  puts ("    isalnum()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (! isalnum (*cp))
      FAIL ("isalnum ('%c') not true", *cp);
  for (cp = upper; *cp != '\0'; ++cp)
    if (! isalnum (*cp))
      FAIL ("isalnum ('%c') not true", *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (! isalnum (*cp))
      FAIL ("isalnum ('%c') not true", *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (isalnum (*cp))
      FAIL ("isalnum ('\\x%02x') is true", *cp);


  puts ("    tolower()");
  for (cp = lower; *cp != '\0'; ++cp)
    if (tolower (*cp) != *cp)
      FAIL ("tolower ('%c') != '%c'", *cp, *cp);
  for (cp = upper, cp2 = lower; *cp != '\0'; ++cp, ++cp2)
    if (tolower (*cp) != *cp2)
      FAIL ("tolower ('%c') != '%c'", *cp, *cp2);
  for (cp = digits; *cp != '\0'; ++cp)
    if (tolower (*cp) != *cp)
      FAIL ("tolower ('%c') != '%c'", *cp, *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (tolower (*cp) != *cp)
      FAIL ("tolower ('\\x%02x') != '\\x%02x'", *cp, *cp);

  puts ("    toupper()");
  for (cp = lower, cp2 = upper; *cp != '\0'; ++cp, ++cp2)
    if (toupper (*cp) != *cp2)
      FAIL ("toupper ('%c') != '%c'", *cp, *cp2);
  for (cp = upper; *cp != '\0'; ++cp)
    if (toupper (*cp) != *cp)
      FAIL ("toupper ('%c') != '%c'", *cp, *cp);
  for (cp = digits; *cp != '\0'; ++cp)
    if (toupper (*cp) != *cp)
      FAIL ("toupper ('%c') != '%c'", *cp, *cp);
  for (cp = cntrl; *cp != '\0'; ++cp)
    if (toupper (*cp) != *cp)
      FAIL ("toupper ('\\x%02x') != '\\x%02x'", *cp, *cp);


  /* Now some locale specific tests.  */
  while (! feof (stdin))
    {
      unsigned char *inp;
      unsigned char *resp;

      if (getline (&inpline, &inplinelen, stdin) <= 0
	  || getline (&resline, &reslinelen, stdin) <= 0)
	break;

      inp = (unsigned char *) strchr (inpline, '\n');
      if (inp != NULL)
	*inp = '\0';
      resp = (unsigned char *) strchr (resline, '\n');
      if (resp != NULL)
	*resp = '\0';

      inp = (unsigned char *) inpline;
      while (*inp != ' ' && *inp != '\t' && *inp && *inp != '\n'
	     && *inp != '\0')
	++inp;

      if (*inp == '\0')
	{
	  printf ("line \"%s\" is without content\n", inpline);
	  continue;
	}
      *inp++ = '\0';
      while (*inp == ' ' || *inp == '\t')
	++inp;

      /* Try all classes.  */
      for (n = 0; n < nclasses; ++n)
	if (strcmp (inpline, classes[n].name) == 0)
	  break;

      resp = (unsigned char *) resline;
      while (*resp == ' ' || *resp == '\t')
	++resp;

      if (strlen ((char *) inp) != strlen ((char *) resp))
	{
	  printf ("lines \"%.20s\"... and \"%.20s\" have not the same length\n",
		  inp, resp);
	  continue;
	}

      if (n < nclasses)
	{
	  if (strspn ((char *) resp, "01") != strlen ((char *) resp))
	    {
	      printf ("result string \"%s\" malformed\n", resp);
	      continue;
	    }

	  printf ("  Locale-specific tests for `%s'\n", inpline);

	  while (*inp != '\0' && *inp != '\n')
	    {
	      if (((__ctype_b[(unsigned int) *inp] & classes[n].mask) != 0)
		  != (*resp != '0'))
		{
		  printf ("    is%s('%c' = '\\x%02x') %s true\n", inpline,
			  *inp, *inp, *resp == '1' ? "not" : "is");
		  ++errors;
		}
	      ++inp;
	      ++resp;
	    }
	}
      else if (strcmp (inpline, "tolower") == 0)
	{
	  while (*inp != '\0')
	    {
	      if (tolower (*inp) != *resp)
		{
		  printf ("    tolower('%c' = '\\x%02x') != '%c'\n",
			  *inp, *inp, *resp);
		  ++errors;
		}
	      ++inp;
	      ++resp;
	    }
	}
      else if (strcmp (inpline, "toupper") == 0)
	{
	  while (*inp != '\0')
	    {
	      if (toupper (*inp) != *resp)
		{
		  printf ("    toupper('%c' = '\\x%02x') != '%c'\n",
			  *inp, *inp, *resp);
		  ++errors;
		}
	      ++inp;
	      ++resp;
	    }
	}
      else
	printf ("\"%s\": unknown class or map\n", inpline);
    }


  if (errors != 0)
    {
      printf ("  %d error%s for `%s' locale\n\n\n", errors,
	      errors == 1 ? "" : "s", setlocale (LC_ALL, NULL));
      return 1;
    }

  printf ("  No errors for `%s' locale\n\n\n", setlocale (LC_ALL, NULL));
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
