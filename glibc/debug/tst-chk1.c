/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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

/* This file tests gets.  Force it to be declared.  */
#include <features.h>
#undef __GLIBC_USE_DEPRECATED_GETS
#define __GLIBC_USE_DEPRECATED_GETS 1

#include <assert.h>
#include <fcntl.h>
#include <locale.h>
#include <obstack.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>
#include <sys/poll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>


#define obstack_chunk_alloc malloc
#define obstack_chunk_free free

char *temp_filename;
static void do_prepare (void);
static int do_test (void);
#define PREPARE(argc, argv) do_prepare ()
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

static void
do_prepare (void)
{
  int temp_fd = create_temp_file ("tst-chk1.", &temp_filename);
  if (temp_fd == -1)
    {
      printf ("cannot create temporary file: %m\n");
      exit (1);
    }

  const char *strs = "abcdefgh\nABCDEFGHI\nabcdefghij\nABCDEFGHIJ";
  if ((size_t) write (temp_fd, strs, strlen (strs)) != strlen (strs))
    {
      puts ("could not write test strings into file");
      unlink (temp_filename);
      exit (1);
    }
}

volatile int chk_fail_ok;
volatile int ret;
jmp_buf chk_fail_buf;

static void
handler (int sig)
{
  if (chk_fail_ok)
    {
      chk_fail_ok = 0;
      longjmp (chk_fail_buf, 1);
    }
  else
    _exit (127);
}

char buf[10];
wchar_t wbuf[10];
volatile size_t l0;
volatile char *p;
volatile wchar_t *wp;
const char *str1 = "JIHGFEDCBA";
const char *str2 = "F";
const char *str3 = "%s%n%s%n";
const char *str4 = "Hello, ";
const char *str5 = "World!\n";
const wchar_t *wstr1 = L"JIHGFEDCBA";
const wchar_t *wstr2 = L"F";
const wchar_t *wstr3 = L"%s%n%s%n";
const wchar_t *wstr4 = L"Hello, ";
const wchar_t *wstr5 = L"World!\n";
char buf2[10] = "%s";
int num1 = 67;
int num2 = 987654;

#define FAIL() \
  do { printf ("Failure on line %d\n", __LINE__); ret = 1; } while (0)
#define CHK_FAIL_START \
  chk_fail_ok = 1;				\
  if (! setjmp (chk_fail_buf))			\
    {
#define CHK_FAIL_END \
      chk_fail_ok = 0;				\
      FAIL ();					\
    }
#if __USE_FORTIFY_LEVEL >= 2 && (!defined __cplusplus || defined __va_arg_pack)
# define CHK_FAIL2_START CHK_FAIL_START
# define CHK_FAIL2_END CHK_FAIL_END
#else
# define CHK_FAIL2_START
# define CHK_FAIL2_END
#endif

static int
do_test (void)
{
  set_fortify_handler (handler);

  struct A { char buf1[9]; char buf2[1]; } a;
  struct wA { wchar_t buf1[9]; wchar_t buf2[1]; } wa;

  printf ("Test checking routines at fortify level %d\n",
#ifdef __USE_FORTIFY_LEVEL
	  (int) __USE_FORTIFY_LEVEL
#else
	  0
#endif
	  );

#if defined __USE_FORTIFY_LEVEL && !defined __fortify_function
  printf ("Test skipped");
  if (l0 == 0)
    return 0;
#endif

  /* These ops can be done without runtime checking of object size.  */
  memcpy (buf, "abcdefghij", 10);
  memmove (buf + 1, buf, 9);
  if (memcmp (buf, "aabcdefghi", 10))
    FAIL ();

  memcpy (buf, "abcdefghij", 10);
  bcopy (buf, buf + 1, 9);
  if (memcmp (buf, "aabcdefghi", 10))
    FAIL ();

  if (mempcpy (buf + 5, "abcde", 5) != buf + 10
      || memcmp (buf, "aabcdabcde", 10))
    FAIL ();

  memset (buf + 8, 'j', 2);
  if (memcmp (buf, "aabcdabcjj", 10))
    FAIL ();

  bzero (buf + 8, 2);
  if (memcmp (buf, "aabcdabc\0\0", 10))
    FAIL ();

  explicit_bzero (buf + 6, 4);
  if (memcmp (buf, "aabcda\0\0\0\0", 10))
    FAIL ();

  strcpy (buf + 4, "EDCBA");
  if (memcmp (buf, "aabcEDCBA", 10))
    FAIL ();

  if (stpcpy (buf + 8, "F") != buf + 9 || memcmp (buf, "aabcEDCBF", 10))
    FAIL ();

  strncpy (buf + 6, "X", 4);
  if (memcmp (buf, "aabcEDX\0\0", 10))
    FAIL ();

  if (sprintf (buf + 7, "%s", "67") != 2 || memcmp (buf, "aabcEDX67", 10))
    FAIL ();

  if (snprintf (buf + 7, 3, "%s", "987654") != 6
      || memcmp (buf, "aabcEDX98", 10))
    FAIL ();

  /* These ops need runtime checking, but shouldn't __chk_fail.  */
  memcpy (buf, "abcdefghij", l0 + 10);
  memmove (buf + 1, buf, l0 + 9);
  if (memcmp (buf, "aabcdefghi", 10))
    FAIL ();

  memcpy (buf, "abcdefghij", l0 + 10);
  bcopy (buf, buf + 1, l0 + 9);
  if (memcmp (buf, "aabcdefghi", 10))
    FAIL ();

  if (mempcpy (buf + 5, "abcde", l0 + 5) != buf + 10
      || memcmp (buf, "aabcdabcde", 10))
    FAIL ();

  memset (buf + 8, 'j', l0 + 2);
  if (memcmp (buf, "aabcdabcjj", 10))
    FAIL ();

  bzero (buf + 8, l0 + 2);
  if (memcmp (buf, "aabcdabc\0\0", 10))
    FAIL ();

  explicit_bzero (buf + 6, l0 + 4);
  if (memcmp (buf, "aabcda\0\0\0\0", 10))
    FAIL ();

  strcpy (buf + 4, str1 + 5);
  if (memcmp (buf, "aabcEDCBA", 10))
    FAIL ();

  if (stpcpy (buf + 8, str2) != buf + 9 || memcmp (buf, "aabcEDCBF", 10))
    FAIL ();

  strncpy (buf + 6, "X", l0 + 4);
  if (memcmp (buf, "aabcEDX\0\0", 10))
    FAIL ();

  if (stpncpy (buf + 5, "cd", l0 + 5) != buf + 7
      || memcmp (buf, "aabcEcd\0\0", 10))
    FAIL ();

  if (sprintf (buf + 7, "%d", num1) != 2 || memcmp (buf, "aabcEcd67", 10))
    FAIL ();

  if (snprintf (buf + 7, 3, "%d", num2) != 6 || memcmp (buf, "aabcEcd98", 10))
    FAIL ();

  buf[l0 + 8] = '\0';
  strcat (buf, "A");
  if (memcmp (buf, "aabcEcd9A", 10))
    FAIL ();

  buf[l0 + 7] = '\0';
  strncat (buf, "ZYXWV", l0 + 2);
  if (memcmp (buf, "aabcEcdZY", 10))
    FAIL ();

  /* The following tests are supposed to succeed at all fortify
     levels, even though they overflow a.buf1 into a.buf2.  */
  memcpy (a.buf1, "abcdefghij", l0 + 10);
  memmove (a.buf1 + 1, a.buf1, l0 + 9);
  if (memcmp (a.buf1, "aabcdefghi", 10))
    FAIL ();

  memcpy (a.buf1, "abcdefghij", l0 + 10);
  bcopy (a.buf1, a.buf1 + 1, l0 + 9);
  if (memcmp (a.buf1, "aabcdefghi", 10))
    FAIL ();

  if (mempcpy (a.buf1 + 5, "abcde", l0 + 5) != a.buf1 + 10
      || memcmp (a.buf1, "aabcdabcde", 10))
    FAIL ();

  memset (a.buf1 + 8, 'j', l0 + 2);
  if (memcmp (a.buf1, "aabcdabcjj", 10))
    FAIL ();

  bzero (a.buf1 + 8, l0 + 2);
  if (memcmp (a.buf1, "aabcdabc\0\0", 10))
    FAIL ();

  explicit_bzero (a.buf1 + 6, l0 + 4);
  if (memcmp (a.buf1, "aabcda\0\0\0\0", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL < 2
  /* The following tests are supposed to crash with -D_FORTIFY_SOURCE=2
     and sufficient GCC support, as the string operations overflow
     from a.buf1 into a.buf2.  */
  strcpy (a.buf1 + 4, str1 + 5);
  if (memcmp (a.buf1, "aabcEDCBA", 10))
    FAIL ();

  if (stpcpy (a.buf1 + 8, str2) != a.buf1 + 9
      || memcmp (a.buf1, "aabcEDCBF", 10))
    FAIL ();

  strncpy (a.buf1 + 6, "X", l0 + 4);
  if (memcmp (a.buf1, "aabcEDX\0\0", 10))
    FAIL ();

  if (sprintf (a.buf1 + 7, "%d", num1) != 2
      || memcmp (a.buf1, "aabcEDX67", 10))
    FAIL ();

  if (snprintf (a.buf1 + 7, 3, "%d", num2) != 6
      || memcmp (a.buf1, "aabcEDX98", 10))
    FAIL ();

  a.buf1[l0 + 8] = '\0';
  strcat (a.buf1, "A");
  if (memcmp (a.buf1, "aabcEDX9A", 10))
    FAIL ();

  a.buf1[l0 + 7] = '\0';
  strncat (a.buf1, "ZYXWV", l0 + 2);
  if (memcmp (a.buf1, "aabcEDXZY", 10))
    FAIL ();

#endif

#if __USE_FORTIFY_LEVEL >= 1
  /* Now check if all buffer overflows are caught at runtime.
     N.B. All tests involving a length parameter need to be done
     twice: once with the length a compile-time constant, once without.  */

  CHK_FAIL_START
  memcpy (buf + 1, "abcdefghij", 10);
  CHK_FAIL_END

  CHK_FAIL_START
  memcpy (buf + 1, "abcdefghij", l0 + 10);
  CHK_FAIL_END

  CHK_FAIL_START
  memmove (buf + 2, buf + 1, 9);
  CHK_FAIL_END

  CHK_FAIL_START
  memmove (buf + 2, buf + 1, l0 + 9);
  CHK_FAIL_END

  CHK_FAIL_START
  bcopy (buf + 1, buf + 2, 9);
  CHK_FAIL_END

  CHK_FAIL_START
  bcopy (buf + 1, buf + 2, l0 + 9);
  CHK_FAIL_END

  CHK_FAIL_START
  p = (char *) mempcpy (buf + 6, "abcde", 5);
  CHK_FAIL_END

  CHK_FAIL_START
  p = (char *) mempcpy (buf + 6, "abcde", l0 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  memset (buf + 9, 'j', 2);
  CHK_FAIL_END

  CHK_FAIL_START
  memset (buf + 9, 'j', l0 + 2);
  CHK_FAIL_END

  CHK_FAIL_START
  bzero (buf + 9, 2);
  CHK_FAIL_END

  CHK_FAIL_START
  bzero (buf + 9, l0 + 2);
  CHK_FAIL_END

  CHK_FAIL_START
  explicit_bzero (buf + 9, 2);
  CHK_FAIL_END

  CHK_FAIL_START
  explicit_bzero (buf + 9, l0 + 2);
  CHK_FAIL_END

  CHK_FAIL_START
  strcpy (buf + 5, str1 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  p = stpcpy (buf + 9, str2);
  CHK_FAIL_END

  CHK_FAIL_START
  strncpy (buf + 7, "X", 4);
  CHK_FAIL_END

  CHK_FAIL_START
  strncpy (buf + 7, "X", l0 + 4);
  CHK_FAIL_END

  CHK_FAIL_START
  stpncpy (buf + 6, "cd", 5);
  CHK_FAIL_END

  CHK_FAIL_START
  stpncpy (buf + 6, "cd", l0 + 5);
  CHK_FAIL_END

# if !defined __cplusplus || defined __va_arg_pack
  CHK_FAIL_START
  sprintf (buf + 8, "%d", num1);
  CHK_FAIL_END

  CHK_FAIL_START
  snprintf (buf + 8, 3, "%d", num2);
  CHK_FAIL_END

  CHK_FAIL_START
  snprintf (buf + 8, l0 + 3, "%d", num2);
  CHK_FAIL_END

  CHK_FAIL_START
  swprintf (wbuf + 8, 3, L"%d", num1);
  CHK_FAIL_END

  CHK_FAIL_START
  swprintf (wbuf + 8, l0 + 3, L"%d", num1);
  CHK_FAIL_END
# endif

  memcpy (buf, str1 + 2, 9);
  CHK_FAIL_START
  strcat (buf, "AB");
  CHK_FAIL_END

  memcpy (buf, str1 + 3, 8);
  CHK_FAIL_START
  strncat (buf, "ZYXWV", 3);
  CHK_FAIL_END

  memcpy (buf, str1 + 3, 8);
  CHK_FAIL_START
  strncat (buf, "ZYXWV", l0 + 3);
  CHK_FAIL_END

  CHK_FAIL_START
  memcpy (a.buf1 + 1, "abcdefghij", 10);
  CHK_FAIL_END

  CHK_FAIL_START
  memcpy (a.buf1 + 1, "abcdefghij", l0 + 10);
  CHK_FAIL_END

  CHK_FAIL_START
  memmove (a.buf1 + 2, a.buf1 + 1, 9);
  CHK_FAIL_END

  CHK_FAIL_START
  memmove (a.buf1 + 2, a.buf1 + 1, l0 + 9);
  CHK_FAIL_END

  CHK_FAIL_START
  bcopy (a.buf1 + 1, a.buf1 + 2, 9);
  CHK_FAIL_END

  CHK_FAIL_START
  bcopy (a.buf1 + 1, a.buf1 + 2, l0 + 9);
  CHK_FAIL_END

  CHK_FAIL_START
  p = (char *) mempcpy (a.buf1 + 6, "abcde", 5);
  CHK_FAIL_END

  CHK_FAIL_START
  p = (char *) mempcpy (a.buf1 + 6, "abcde", l0 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  memset (a.buf1 + 9, 'j', 2);
  CHK_FAIL_END

  CHK_FAIL_START
  memset (a.buf1 + 9, 'j', l0 + 2);
  CHK_FAIL_END

  CHK_FAIL_START
  bzero (a.buf1 + 9, 2);
  CHK_FAIL_END

  CHK_FAIL_START
  bzero (a.buf1 + 9, l0 + 2);
  CHK_FAIL_END

  CHK_FAIL_START
  explicit_bzero (a.buf1 + 9, 2);
  CHK_FAIL_END

  CHK_FAIL_START
  explicit_bzero (a.buf1 + 9, l0 + 2);
  CHK_FAIL_END

# if __USE_FORTIFY_LEVEL >= 2
#  define O 0
# else
#  define O 1
# endif

  CHK_FAIL_START
  strcpy (a.buf1 + (O + 4), str1 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  p = stpcpy (a.buf1 + (O + 8), str2);
  CHK_FAIL_END

  CHK_FAIL_START
  strncpy (a.buf1 + (O + 6), "X", 4);
  CHK_FAIL_END

  CHK_FAIL_START
  strncpy (a.buf1 + (O + 6), "X", l0 + 4);
  CHK_FAIL_END

# if !defined __cplusplus || defined __va_arg_pack
  CHK_FAIL_START
  sprintf (a.buf1 + (O + 7), "%d", num1);
  CHK_FAIL_END

  CHK_FAIL_START
  snprintf (a.buf1 + (O + 7), 3, "%d", num2);
  CHK_FAIL_END

  CHK_FAIL_START
  snprintf (a.buf1 + (O + 7), l0 + 3, "%d", num2);
  CHK_FAIL_END
# endif

  memcpy (a.buf1, str1 + (3 - O), 8 + O);
  CHK_FAIL_START
  strcat (a.buf1, "AB");
  CHK_FAIL_END

  memcpy (a.buf1, str1 + (4 - O), 7 + O);
  CHK_FAIL_START
  strncat (a.buf1, "ZYXWV", l0 + 3);
  CHK_FAIL_END
#endif


  /* These ops can be done without runtime checking of object size.  */
  wmemcpy (wbuf, L"abcdefghij", 10);
  wmemmove (wbuf + 1, wbuf, 9);
  if (wmemcmp (wbuf, L"aabcdefghi", 10))
    FAIL ();

  if (wmempcpy (wbuf + 5, L"abcde", 5) != wbuf + 10
      || wmemcmp (wbuf, L"aabcdabcde", 10))
    FAIL ();

  wmemset (wbuf + 8, L'j', 2);
  if (wmemcmp (wbuf, L"aabcdabcjj", 10))
    FAIL ();

  wcscpy (wbuf + 4, L"EDCBA");
  if (wmemcmp (wbuf, L"aabcEDCBA", 10))
    FAIL ();

  if (wcpcpy (wbuf + 8, L"F") != wbuf + 9 || wmemcmp (wbuf, L"aabcEDCBF", 10))
    FAIL ();

  wcsncpy (wbuf + 6, L"X", 4);
  if (wmemcmp (wbuf, L"aabcEDX\0\0", 10))
    FAIL ();

  if (swprintf (wbuf + 7, 3, L"%ls", L"987654") >= 0
      || wmemcmp (wbuf, L"aabcEDX98", 10))
    FAIL ();

  if (swprintf (wbuf + 7, 3, L"64") != 2
      || wmemcmp (wbuf, L"aabcEDX64", 10))
    FAIL ();

  /* These ops need runtime checking, but shouldn't __chk_fail.  */
  wmemcpy (wbuf, L"abcdefghij", l0 + 10);
  wmemmove (wbuf + 1, wbuf, l0 + 9);
  if (wmemcmp (wbuf, L"aabcdefghi", 10))
    FAIL ();

  if (wmempcpy (wbuf + 5, L"abcde", l0 + 5) != wbuf + 10
      || wmemcmp (wbuf, L"aabcdabcde", 10))
    FAIL ();

  wmemset (wbuf + 8, L'j', l0 + 2);
  if (wmemcmp (wbuf, L"aabcdabcjj", 10))
    FAIL ();

  wcscpy (wbuf + 4, wstr1 + 5);
  if (wmemcmp (wbuf, L"aabcEDCBA", 10))
    FAIL ();

  if (wcpcpy (wbuf + 8, wstr2) != wbuf + 9 || wmemcmp (wbuf, L"aabcEDCBF", 10))
    FAIL ();

  wcsncpy (wbuf + 6, L"X", l0 + 4);
  if (wmemcmp (wbuf, L"aabcEDX\0\0", 10))
    FAIL ();

  if (wcpncpy (wbuf + 5, L"cd", l0 + 5) != wbuf + 7
      || wmemcmp (wbuf, L"aabcEcd\0\0", 10))
    FAIL ();

  if (swprintf (wbuf + 7, 3, L"%d", num2) >= 0
      || wmemcmp (wbuf, L"aabcEcd98", 10))
    FAIL ();

  wbuf[l0 + 8] = L'\0';
  wcscat (wbuf, L"A");
  if (wmemcmp (wbuf, L"aabcEcd9A", 10))
    FAIL ();

  wbuf[l0 + 7] = L'\0';
  wcsncat (wbuf, L"ZYXWV", l0 + 2);
  if (wmemcmp (wbuf, L"aabcEcdZY", 10))
    FAIL ();

  wmemcpy (wa.buf1, L"abcdefghij", l0 + 10);
  wmemmove (wa.buf1 + 1, wa.buf1, l0 + 9);
  if (wmemcmp (wa.buf1, L"aabcdefghi", 10))
    FAIL ();

  if (wmempcpy (wa.buf1 + 5, L"abcde", l0 + 5) != wa.buf1 + 10
      || wmemcmp (wa.buf1, L"aabcdabcde", 10))
    FAIL ();

  wmemset (wa.buf1 + 8, L'j', l0 + 2);
  if (wmemcmp (wa.buf1, L"aabcdabcjj", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL < 2
  /* The following tests are supposed to crash with -D_FORTIFY_SOURCE=2
     and sufficient GCC support, as the string operations overflow
     from a.buf1 into a.buf2.  */
  wcscpy (wa.buf1 + 4, wstr1 + 5);
  if (wmemcmp (wa.buf1, L"aabcEDCBA", 10))
    FAIL ();

  if (wcpcpy (wa.buf1 + 8, wstr2) != wa.buf1 + 9
      || wmemcmp (wa.buf1, L"aabcEDCBF", 10))
    FAIL ();

  wcsncpy (wa.buf1 + 6, L"X", l0 + 4);
  if (wmemcmp (wa.buf1, L"aabcEDX\0\0", 10))
    FAIL ();

  if (swprintf (wa.buf1 + 7, 3, L"%d", num2) >= 0
      || wmemcmp (wa.buf1, L"aabcEDX98", 10))
    FAIL ();

  wa.buf1[l0 + 8] = L'\0';
  wcscat (wa.buf1, L"A");
  if (wmemcmp (wa.buf1, L"aabcEDX9A", 10))
    FAIL ();

  wa.buf1[l0 + 7] = L'\0';
  wcsncat (wa.buf1, L"ZYXWV", l0 + 2);
  if (wmemcmp (wa.buf1, L"aabcEDXZY", 10))
    FAIL ();

#endif

#if __USE_FORTIFY_LEVEL >= 1
  /* Now check if all buffer overflows are caught at runtime.
     N.B. All tests involving a length parameter need to be done
     twice: once with the length a compile-time constant, once without.  */

  CHK_FAIL_START
  wmemcpy (wbuf + 1, L"abcdefghij", 10);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemcpy (wbuf + 1, L"abcdefghij", l0 + 10);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemcpy (wbuf + 9, L"abcdefghij", 10);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemcpy (wbuf + 9, L"abcdefghij", l0 + 10);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemmove (wbuf + 2, wbuf + 1, 9);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemmove (wbuf + 2, wbuf + 1, l0 + 9);
  CHK_FAIL_END

  CHK_FAIL_START
  wp = wmempcpy (wbuf + 6, L"abcde", 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wp = wmempcpy (wbuf + 6, L"abcde", l0 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemset (wbuf + 9, L'j', 2);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemset (wbuf + 9, L'j', l0 + 2);
  CHK_FAIL_END

  CHK_FAIL_START
  wcscpy (wbuf + 5, wstr1 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wp = wcpcpy (wbuf + 9, wstr2);
  CHK_FAIL_END

  CHK_FAIL_START
  wcsncpy (wbuf + 7, L"X", 4);
  CHK_FAIL_END

  CHK_FAIL_START
  wcsncpy (wbuf + 7, L"X", l0 + 4);
  CHK_FAIL_END

  CHK_FAIL_START
  wcsncpy (wbuf + 9, L"XABCDEFGH", 8);
  CHK_FAIL_END

  CHK_FAIL_START
  wcpncpy (wbuf + 9, L"XABCDEFGH", 8);
  CHK_FAIL_END

  CHK_FAIL_START
  wcpncpy (wbuf + 6, L"cd", 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wcpncpy (wbuf + 6, L"cd", l0 + 5);
  CHK_FAIL_END

  wmemcpy (wbuf, wstr1 + 2, 9);
  CHK_FAIL_START
  wcscat (wbuf, L"AB");
  CHK_FAIL_END

  wmemcpy (wbuf, wstr1 + 3, 8);
  CHK_FAIL_START
  wcsncat (wbuf, L"ZYXWV", l0 + 3);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemcpy (wa.buf1 + 1, L"abcdefghij", 10);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemcpy (wa.buf1 + 1, L"abcdefghij", l0 + 10);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemmove (wa.buf1 + 2, wa.buf1 + 1, 9);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemmove (wa.buf1 + 2, wa.buf1 + 1, l0 + 9);
  CHK_FAIL_END

  CHK_FAIL_START
  wp = wmempcpy (wa.buf1 + 6, L"abcde", 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wp = wmempcpy (wa.buf1 + 6, L"abcde", l0 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemset (wa.buf1 + 9, L'j', 2);
  CHK_FAIL_END

  CHK_FAIL_START
  wmemset (wa.buf1 + 9, L'j', l0 + 2);
  CHK_FAIL_END

#if __USE_FORTIFY_LEVEL >= 2
# define O 0
#else
# define O 1
#endif

  CHK_FAIL_START
  wcscpy (wa.buf1 + (O + 4), wstr1 + 5);
  CHK_FAIL_END

  CHK_FAIL_START
  wp = wcpcpy (wa.buf1 + (O + 8), wstr2);
  CHK_FAIL_END

  CHK_FAIL_START
  wcsncpy (wa.buf1 + (O + 6), L"X", 4);
  CHK_FAIL_END

  CHK_FAIL_START
  wcsncpy (wa.buf1 + (O + 6), L"X", l0 + 4);
  CHK_FAIL_END

  wmemcpy (wa.buf1, wstr1 + (3 - O), 8 + O);
  CHK_FAIL_START
  wcscat (wa.buf1, L"AB");
  CHK_FAIL_END

  wmemcpy (wa.buf1, wstr1 + (4 - O), 7 + O);
  CHK_FAIL_START
  wcsncat (wa.buf1, L"ZYXWV", l0 + 3);
  CHK_FAIL_END
#endif


  /* Now checks for %n protection.  */

  /* Constant literals passed directly are always ok
     (even with warnings about possible bugs from GCC).  */
  int n1, n2;
  if (sprintf (buf, "%s%n%s%n", str2, &n1, str2, &n2) != 2
      || n1 != 1 || n2 != 2)
    FAIL ();

  /* In this case the format string is not known at compile time,
     but resides in read-only memory, so is ok.  */
  if (snprintf (buf, 4, str3, str2, &n1, str2, &n2) != 2
      || n1 != 1 || n2 != 2)
    FAIL ();

  strcpy (buf2 + 2, "%n%s%n");
  /* When the format string is writable and contains %n,
     with -D_FORTIFY_SOURCE=2 it causes __chk_fail.  */
  CHK_FAIL2_START
  if (sprintf (buf, buf2, str2, &n1, str2, &n1) != 2)
    FAIL ();
  CHK_FAIL2_END

  CHK_FAIL2_START
  if (snprintf (buf, 3, buf2, str2, &n1, str2, &n1) != 2)
    FAIL ();
  CHK_FAIL2_END

  /* But if there is no %n, even writable format string
     should work.  */
  buf2[6] = '\0';
  if (sprintf (buf, buf2 + 4, str2) != 1)
    FAIL ();

  /* Constant literals passed directly are always ok
     (even with warnings about possible bugs from GCC).  */
  if (printf ("%s%n%s%n", str4, &n1, str5, &n2) != 14
      || n1 != 7 || n2 != 14)
    FAIL ();

  /* In this case the format string is not known at compile time,
     but resides in read-only memory, so is ok.  */
  if (printf (str3, str4, &n1, str5, &n2) != 14
      || n1 != 7 || n2 != 14)
    FAIL ();

  strcpy (buf2 + 2, "%n%s%n");
  /* When the format string is writable and contains %n,
     with -D_FORTIFY_SOURCE=2 it causes __chk_fail.  */
  CHK_FAIL2_START
  if (printf (buf2, str4, &n1, str5, &n1) != 14)
    FAIL ();
  CHK_FAIL2_END

  /* But if there is no %n, even writable format string
     should work.  */
  buf2[6] = '\0';
  if (printf (buf2 + 4, str5) != 7)
    FAIL ();

  FILE *fp = stdout;

  /* Constant literals passed directly are always ok
     (even with warnings about possible bugs from GCC).  */
  if (fprintf (fp, "%s%n%s%n", str4, &n1, str5, &n2) != 14
      || n1 != 7 || n2 != 14)
    FAIL ();

  /* In this case the format string is not known at compile time,
     but resides in read-only memory, so is ok.  */
  if (fprintf (fp, str3, str4, &n1, str5, &n2) != 14
      || n1 != 7 || n2 != 14)
    FAIL ();

  strcpy (buf2 + 2, "%n%s%n");
  /* When the format string is writable and contains %n,
     with -D_FORTIFY_SOURCE=2 it causes __chk_fail.  */
  CHK_FAIL2_START
  if (fprintf (fp, buf2, str4, &n1, str5, &n1) != 14)
    FAIL ();
  CHK_FAIL2_END

  /* But if there is no %n, even writable format string
     should work.  */
  buf2[6] = '\0';
  if (fprintf (fp, buf2 + 4, str5) != 7)
    FAIL ();

  char *my_ptr = NULL;
  strcpy (buf2 + 2, "%n%s%n");
  /* When the format string is writable and contains %n,
     with -D_FORTIFY_SOURCE=2 it causes __chk_fail.  */
  CHK_FAIL2_START
  if (asprintf (&my_ptr, buf2, str4, &n1, str5, &n1) != 14)
    FAIL ();
  else
    free (my_ptr);
  CHK_FAIL2_END

  struct obstack obs;
  obstack_init (&obs);
  CHK_FAIL2_START
  if (obstack_printf (&obs, buf2, str4, &n1, str5, &n1) != 14)
    FAIL ();
  CHK_FAIL2_END
  obstack_free (&obs, NULL);

  my_ptr = NULL;
  if (asprintf (&my_ptr, "%s%n%s%n", str4, &n1, str5, &n1) != 14)
    FAIL ();
  else
    free (my_ptr);

  obstack_init (&obs);
  if (obstack_printf (&obs, "%s%n%s%n", str4, &n1, str5, &n1) != 14)
    FAIL ();
  obstack_free (&obs, NULL);

  if (freopen (temp_filename, "r", stdin) == NULL)
    {
      puts ("could not open temporary file");
      exit (1);
    }

  if (gets (buf) != buf || memcmp (buf, "abcdefgh", 9))
    FAIL ();
  if (gets (buf) != buf || memcmp (buf, "ABCDEFGHI", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (gets (buf) != buf)
    FAIL ();
  CHK_FAIL_END
#endif

  rewind (stdin);

  if (fgets (buf, sizeof (buf), stdin) != buf
      || memcmp (buf, "abcdefgh\n", 10))
    FAIL ();
  if (fgets (buf, sizeof (buf), stdin) != buf || memcmp (buf, "ABCDEFGHI", 10))
    FAIL ();

  rewind (stdin);

  if (fgets (buf, l0 + sizeof (buf), stdin) != buf
      || memcmp (buf, "abcdefgh\n", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (fgets (buf, sizeof (buf) + 1, stdin) != buf)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (fgets (buf, l0 + sizeof (buf) + 1, stdin) != buf)
    FAIL ();
  CHK_FAIL_END
#endif

  rewind (stdin);

  if (fgets_unlocked (buf, sizeof (buf), stdin) != buf
      || memcmp (buf, "abcdefgh\n", 10))
    FAIL ();
  if (fgets_unlocked (buf, sizeof (buf), stdin) != buf
      || memcmp (buf, "ABCDEFGHI", 10))
    FAIL ();

  rewind (stdin);

  if (fgets_unlocked (buf, l0 + sizeof (buf), stdin) != buf
      || memcmp (buf, "abcdefgh\n", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (fgets_unlocked (buf, sizeof (buf) + 1, stdin) != buf)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (fgets_unlocked (buf, l0 + sizeof (buf) + 1, stdin) != buf)
    FAIL ();
  CHK_FAIL_END
#endif

  rewind (stdin);

  if (fread (buf, 1, sizeof (buf), stdin) != sizeof (buf)
      || memcmp (buf, "abcdefgh\nA", 10))
    FAIL ();
  if (fread (buf, sizeof (buf), 1, stdin) != 1
      || memcmp (buf, "BCDEFGHI\na", 10))
    FAIL ();

  rewind (stdin);

  if (fread (buf, l0 + 1, sizeof (buf), stdin) != sizeof (buf)
      || memcmp (buf, "abcdefgh\nA", 10))
    FAIL ();
  if (fread (buf, sizeof (buf), l0 + 1, stdin) != 1
      || memcmp (buf, "BCDEFGHI\na", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (fread (buf, 1, sizeof (buf) + 1, stdin) != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (fread (buf, sizeof (buf) + 1, l0 + 1, stdin) != 1)
    FAIL ();
  CHK_FAIL_END
#endif

  rewind (stdin);

  if (fread_unlocked (buf, 1, sizeof (buf), stdin) != sizeof (buf)
      || memcmp (buf, "abcdefgh\nA", 10))
    FAIL ();
  if (fread_unlocked (buf, sizeof (buf), 1, stdin) != 1
      || memcmp (buf, "BCDEFGHI\na", 10))
    FAIL ();

  rewind (stdin);

  if (fread_unlocked (buf, 1, 4, stdin) != 4
      || memcmp (buf, "abcdFGHI\na", 10))
    FAIL ();
  if (fread_unlocked (buf, 4, 1, stdin) != 1
      || memcmp (buf, "efghFGHI\na", 10))
    FAIL ();

  rewind (stdin);

  if (fread_unlocked (buf, l0 + 1, sizeof (buf), stdin) != sizeof (buf)
      || memcmp (buf, "abcdefgh\nA", 10))
    FAIL ();
  if (fread_unlocked (buf, sizeof (buf), l0 + 1, stdin) != 1
      || memcmp (buf, "BCDEFGHI\na", 10))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (fread_unlocked (buf, 1, sizeof (buf) + 1, stdin) != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (fread_unlocked (buf, sizeof (buf) + 1, l0 + 1, stdin) != 1)
    FAIL ();
  CHK_FAIL_END
#endif

  lseek (fileno (stdin), 0, SEEK_SET);

  if (read (fileno (stdin), buf, sizeof (buf) - 1) != sizeof (buf) - 1
      || memcmp (buf, "abcdefgh\n", 9))
    FAIL ();
  if (read (fileno (stdin), buf, sizeof (buf) - 1) != sizeof (buf) - 1
      || memcmp (buf, "ABCDEFGHI", 9))
    FAIL ();

  lseek (fileno (stdin), 0, SEEK_SET);

  if (read (fileno (stdin), buf, l0 + sizeof (buf) - 1) != sizeof (buf) - 1
      || memcmp (buf, "abcdefgh\n", 9))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (read (fileno (stdin), buf, sizeof (buf) + 1) != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (read (fileno (stdin), buf, l0 + sizeof (buf) + 1) != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END
#endif

  if (pread (fileno (stdin), buf, sizeof (buf) - 1, sizeof (buf) - 2)
      != sizeof (buf) - 1
      || memcmp (buf, "\nABCDEFGH", 9))
    FAIL ();
  if (pread (fileno (stdin), buf, sizeof (buf) - 1, 0) != sizeof (buf) - 1
      || memcmp (buf, "abcdefgh\n", 9))
    FAIL ();
  if (pread (fileno (stdin), buf, l0 + sizeof (buf) - 1, sizeof (buf) - 3)
      != sizeof (buf) - 1
      || memcmp (buf, "h\nABCDEFG", 9))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (pread (fileno (stdin), buf, sizeof (buf) + 1, 2 * sizeof (buf))
      != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (pread (fileno (stdin), buf, l0 + sizeof (buf) + 1, 2 * sizeof (buf))
      != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END
#endif

  if (pread64 (fileno (stdin), buf, sizeof (buf) - 1, sizeof (buf) - 2)
      != sizeof (buf) - 1
      || memcmp (buf, "\nABCDEFGH", 9))
    FAIL ();
  if (pread64 (fileno (stdin), buf, sizeof (buf) - 1, 0) != sizeof (buf) - 1
      || memcmp (buf, "abcdefgh\n", 9))
    FAIL ();
  if (pread64 (fileno (stdin), buf, l0 + sizeof (buf) - 1, sizeof (buf) - 3)
      != sizeof (buf) - 1
      || memcmp (buf, "h\nABCDEFG", 9))
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (pread64 (fileno (stdin), buf, sizeof (buf) + 1, 2 * sizeof (buf))
      != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (pread64 (fileno (stdin), buf, l0 + sizeof (buf) + 1, 2 * sizeof (buf))
      != sizeof (buf) + 1)
    FAIL ();
  CHK_FAIL_END
#endif

  if (freopen (temp_filename, "r", stdin) == NULL)
    {
      puts ("could not open temporary file");
      exit (1);
    }

  if (fseek (stdin, 9 + 10 + 11, SEEK_SET))
    {
      puts ("could not seek in test file");
      exit (1);
    }

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (gets (buf) != buf)
    FAIL ();
  CHK_FAIL_END
#endif

  /* Check whether missing N$ formats are detected.  */
  CHK_FAIL2_START
  printf ("%3$d\n", 1, 2, 3, 4);
  CHK_FAIL2_END

  CHK_FAIL2_START
  fprintf (stdout, "%3$d\n", 1, 2, 3, 4);
  CHK_FAIL2_END

  CHK_FAIL2_START
  sprintf (buf, "%3$d\n", 1, 2, 3, 4);
  CHK_FAIL2_END

  CHK_FAIL2_START
  snprintf (buf, sizeof (buf), "%3$d\n", 1, 2, 3, 4);
  CHK_FAIL2_END

  int sp[2];
  if (socketpair (PF_UNIX, SOCK_STREAM, 0, sp))
    FAIL ();
  else
    {
      const char *sendstr = "abcdefgh\nABCDEFGH\n0123456789\n";
      if ((size_t) send (sp[0], sendstr, strlen (sendstr), 0)
	  != strlen (sendstr))
	FAIL ();

      char recvbuf[12];
      if (recv (sp[1], recvbuf, sizeof recvbuf, MSG_PEEK)
	  != sizeof recvbuf
	  || memcmp (recvbuf, sendstr, sizeof recvbuf) != 0)
	FAIL ();

      if (recv (sp[1], recvbuf + 6, l0 + sizeof recvbuf - 7, MSG_PEEK)
	  != sizeof recvbuf - 7
	  || memcmp (recvbuf + 6, sendstr, sizeof recvbuf - 7) != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      if (recv (sp[1], recvbuf + 1, sizeof recvbuf, MSG_PEEK)
	  != sizeof recvbuf)
	FAIL ();
      CHK_FAIL_END

      CHK_FAIL_START
      if (recv (sp[1], recvbuf + 4, l0 + sizeof recvbuf - 3, MSG_PEEK)
	  != sizeof recvbuf - 3)
	FAIL ();
      CHK_FAIL_END
#endif

      socklen_t sl;
      struct sockaddr_un sa_un;

      sl = sizeof (sa_un);
      if (recvfrom (sp[1], recvbuf, sizeof recvbuf, MSG_PEEK,
		    (struct sockaddr *) &sa_un, &sl)
	  != sizeof recvbuf
	  || memcmp (recvbuf, sendstr, sizeof recvbuf) != 0)
	FAIL ();

      sl = sizeof (sa_un);
      if (recvfrom (sp[1], recvbuf + 6, l0 + sizeof recvbuf - 7, MSG_PEEK,
		    (struct sockaddr *) &sa_un, &sl) != sizeof recvbuf - 7
	  || memcmp (recvbuf + 6, sendstr, sizeof recvbuf - 7) != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      sl = sizeof (sa_un);
      if (recvfrom (sp[1], recvbuf + 1, sizeof recvbuf, MSG_PEEK,
		    (struct sockaddr *) &sa_un, &sl) != sizeof recvbuf)
	FAIL ();
      CHK_FAIL_END

      CHK_FAIL_START
      sl = sizeof (sa_un);
      if (recvfrom (sp[1], recvbuf + 4, l0 + sizeof recvbuf - 3, MSG_PEEK,
		    (struct sockaddr *) &sa_un, &sl) != sizeof recvbuf - 3)
	FAIL ();
      CHK_FAIL_END
#endif

      close (sp[0]);
      close (sp[1]);
    }

  char fname[] = "/tmp/tst-chk1-dir-XXXXXX\0foo";
  char *enddir = strchr (fname, '\0');
  if (mkdtemp (fname) == NULL)
    {
      printf ("mkdtemp failed: %m\n");
      return 1;
    }
  *enddir = '/';
  if (symlink ("bar", fname) != 0)
    FAIL ();

  char readlinkbuf[4];
  if (readlink (fname, readlinkbuf, 4) != 3
      || memcmp (readlinkbuf, "bar", 3) != 0)
    FAIL ();
  if (readlink (fname, readlinkbuf + 1, l0 + 3) != 3
      || memcmp (readlinkbuf, "bbar", 4) != 0)
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (readlink (fname, readlinkbuf + 2, l0 + 3) != 3)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (readlink (fname, readlinkbuf + 3, 4) != 3)
    FAIL ();
  CHK_FAIL_END
#endif

  int tmpfd = open ("/tmp", O_RDONLY | O_DIRECTORY);
  if (tmpfd < 0)
    FAIL ();

  if (readlinkat (tmpfd, fname + sizeof ("/tmp/") - 1, readlinkbuf, 4) != 3
      || memcmp (readlinkbuf, "bar", 3) != 0)
    FAIL ();
  if (readlinkat (tmpfd, fname + sizeof ("/tmp/") - 1, readlinkbuf + 1,
		  l0 + 3) != 3
      || memcmp (readlinkbuf, "bbar", 4) != 0)
    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  if (readlinkat (tmpfd, fname + sizeof ("/tmp/") - 1, readlinkbuf + 2,
		  l0 + 3) != 3)
    FAIL ();
  CHK_FAIL_END

  CHK_FAIL_START
  if (readlinkat (tmpfd, fname + sizeof ("/tmp/") - 1, readlinkbuf + 3,
		  4) != 3)
    FAIL ();
  CHK_FAIL_END
#endif

  close (tmpfd);

  char *cwd1 = getcwd (NULL, 0);
  if (cwd1 == NULL)
    FAIL ();

  char *cwd2 = getcwd (NULL, 250);
  if (cwd2 == NULL)
    FAIL ();

  if (cwd1 && cwd2)
    {
      if (strcmp (cwd1, cwd2) != 0)
	FAIL ();

      *enddir = '\0';
      if (chdir (fname))
	FAIL ();

      char *cwd3 = getcwd (NULL, 0);
      if (cwd3 == NULL)
	FAIL ();
      if (strcmp (fname, cwd3) != 0)
	printf ("getcwd after chdir is '%s' != '%s',"
		"get{c,}wd tests skipped\n", cwd3, fname);
      else
	{
	  char getcwdbuf[sizeof fname - 3];

	  char *cwd4 = getcwd (getcwdbuf, sizeof getcwdbuf);
	  if (cwd4 != getcwdbuf
	      || strcmp (getcwdbuf, fname) != 0)
	    FAIL ();

	  cwd4 = getcwd (getcwdbuf + 1, l0 + sizeof getcwdbuf - 1);
	  if (cwd4 != getcwdbuf + 1
	      || getcwdbuf[0] != fname[0]
	      || strcmp (getcwdbuf + 1, fname) != 0)
	    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
	  CHK_FAIL_START
	  if (getcwd (getcwdbuf + 2, l0 + sizeof getcwdbuf)
	      != getcwdbuf + 2)
	    FAIL ();
	  CHK_FAIL_END

	  CHK_FAIL_START
	  if (getcwd (getcwdbuf + 2, sizeof getcwdbuf)
	      != getcwdbuf + 2)
	    FAIL ();
	  CHK_FAIL_END
#endif

	  if (getwd (getcwdbuf) != getcwdbuf
	      || strcmp (getcwdbuf, fname) != 0)
	    FAIL ();

	  if (getwd (getcwdbuf + 1) != getcwdbuf + 1
	      || strcmp (getcwdbuf + 1, fname) != 0)
	    FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
	  CHK_FAIL_START
	  if (getwd (getcwdbuf + 2) != getcwdbuf + 2)
	    FAIL ();
	  CHK_FAIL_END
#endif
	}

      if (chdir (cwd1) != 0)
	FAIL ();
      free (cwd3);
    }

  free (cwd1);
  free (cwd2);
  *enddir = '/';
  if (unlink (fname) != 0)
    FAIL ();

  *enddir = '\0';
  if (rmdir (fname) != 0)
    FAIL ();


#if PATH_MAX > 0
  char largebuf[PATH_MAX];
  char *realres = realpath (".", largebuf);
  if (realres != largebuf)
    FAIL ();

# if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  char realbuf[1];
  realres = realpath (".", realbuf);
  if (realres != realbuf)
    FAIL ();
  CHK_FAIL_END
# endif
#endif

  if (setlocale (LC_ALL, "de_DE.UTF-8") != NULL)
    {
      assert (MB_CUR_MAX <= 10);

      /* First a simple test.  */
      char enough[10];
      if (wctomb (enough, L'A') != 1)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      /* We know the wchar_t encoding is ISO 10646.  So pick a
	 character which has a multibyte representation which does not
	 fit.  */
      CHK_FAIL_START
      char smallbuf[2];
      if (wctomb (smallbuf, L'\x100') != 2)
	FAIL ();
      CHK_FAIL_END
#endif

      mbstate_t s;
      memset (&s, '\0', sizeof (s));
      if (wcrtomb (enough, L'D', &s) != 1 || enough[0] != 'D')
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      /* We know the wchar_t encoding is ISO 10646.  So pick a
	 character which has a multibyte representation which does not
	 fit.  */
      CHK_FAIL_START
      char smallbuf[2];
      if (wcrtomb (smallbuf, L'\x100', &s) != 2)
	FAIL ();
      CHK_FAIL_END
#endif

      wchar_t wenough[10];
      memset (&s, '\0', sizeof (s));
      const char *cp = "A";
      if (mbsrtowcs (wenough, &cp, 10, &s) != 1
	  || wcscmp (wenough, L"A") != 0)
	FAIL ();

      cp = "BC";
      if (mbsrtowcs (wenough, &cp, l0 + 10, &s) != 2
	  || wcscmp (wenough, L"BC") != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      wchar_t wsmallbuf[2];
      cp = "ABC";
      mbsrtowcs (wsmallbuf, &cp, 10, &s);
      CHK_FAIL_END
#endif

      cp = "A";
      if (mbstowcs (wenough, cp, 10) != 1
	  || wcscmp (wenough, L"A") != 0)
	FAIL ();

      cp = "DEF";
      if (mbstowcs (wenough, cp, l0 + 10) != 3
	  || wcscmp (wenough, L"DEF") != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      wchar_t wsmallbuf[2];
      cp = "ABC";
      mbstowcs (wsmallbuf, cp, 10);
      CHK_FAIL_END
#endif

      memset (&s, '\0', sizeof (s));
      cp = "ABC";
      wcscpy (wenough, L"DEF");
      if (mbsnrtowcs (wenough, &cp, 1, 10, &s) != 1
	  || wcscmp (wenough, L"AEF") != 0)
	FAIL ();

      cp = "IJ";
      if (mbsnrtowcs (wenough, &cp, 1, l0 + 10, &s) != 1
	  || wcscmp (wenough, L"IEF") != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      wchar_t wsmallbuf[2];
      cp = "ABC";
      mbsnrtowcs (wsmallbuf, &cp, 3, 10, &s);
      CHK_FAIL_END
#endif

      memset (&s, '\0', sizeof (s));
      const wchar_t *wcp = L"A";
      if (wcsrtombs (enough, &wcp, 10, &s) != 1
	  || strcmp (enough, "A") != 0)
	FAIL ();

      wcp = L"BC";
      if (wcsrtombs (enough, &wcp, l0 + 10, &s) != 2
	  || strcmp (enough, "BC") != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      char smallbuf[2];
      wcp = L"ABC";
      wcsrtombs (smallbuf, &wcp, 10, &s);
      CHK_FAIL_END
#endif

      memset (enough, 'Z', sizeof (enough));
      wcp = L"EF";
      if (wcstombs (enough, wcp, 10) != 2
	  || strcmp (enough, "EF") != 0)
	FAIL ();

      wcp = L"G";
      if (wcstombs (enough, wcp, l0 + 10) != 1
	  || strcmp (enough, "G") != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      char smallbuf[2];
      wcp = L"ABC";
      wcstombs (smallbuf, wcp, 10);
      CHK_FAIL_END
#endif

      memset (&s, '\0', sizeof (s));
      wcp = L"AB";
      if (wcsnrtombs (enough, &wcp, 1, 10, &s) != 1
	  || strcmp (enough, "A") != 0)
	FAIL ();

      wcp = L"BCD";
      if (wcsnrtombs (enough, &wcp, 1, l0 + 10, &s) != 1
	  || strcmp (enough, "B") != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      char smallbuf[2];
      wcp = L"ABC";
      wcsnrtombs (smallbuf, &wcp, 3, 10, &s);
      CHK_FAIL_END
#endif
    }
  else
    {
      puts ("cannot set locale");
      ret = 1;
    }

  int fd = posix_openpt (O_RDWR);
  if (fd != -1)
    {
      char enough[1000];
      if (ptsname_r (fd, enough, sizeof (enough)) != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      char smallbuf[2];
      if (ptsname_r (fd, smallbuf, sizeof (smallbuf) + 1) == 0)
	FAIL ();
      CHK_FAIL_END
#endif
      close (fd);
    }

#if PATH_MAX > 0
  confstr (_CS_GNU_LIBC_VERSION, largebuf, sizeof (largebuf));
# if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  char smallbuf[1];
  confstr (_CS_GNU_LIBC_VERSION, smallbuf, sizeof (largebuf));
  CHK_FAIL_END
# endif
#endif

  gid_t grpslarge[5];
  int ngr = getgroups (5, grpslarge);
  asm volatile ("" : : "r" (ngr));
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  char smallbuf[1];
  ngr = getgroups (5, (gid_t *) smallbuf);
  asm volatile ("" : : "r" (ngr));
  CHK_FAIL_END
#endif

  fd = open (_PATH_TTY, O_RDONLY);
  if (fd != -1)
    {
      char enough[1000];
      if (ttyname_r (fd, enough, sizeof (enough)) != 0)
	FAIL ();

#if __USE_FORTIFY_LEVEL >= 1
      CHK_FAIL_START
      char smallbuf[2];
      if (ttyname_r (fd, smallbuf, sizeof (smallbuf) + 1) == 0)
	FAIL ();
      CHK_FAIL_END
#endif
      close (fd);
    }

  char hostnamelarge[1000];
  gethostname (hostnamelarge, sizeof (hostnamelarge));
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  char smallbuf[1];
  gethostname (smallbuf, sizeof (hostnamelarge));
  CHK_FAIL_END
#endif

  char loginlarge[1000];
  getlogin_r (loginlarge, sizeof (hostnamelarge));
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  char smallbuf[1];
  getlogin_r (smallbuf, sizeof (loginlarge));
  CHK_FAIL_END
#endif

  char domainnamelarge[1000];
  int res = getdomainname (domainnamelarge, sizeof (domainnamelarge));
  asm volatile ("" : : "r" (res));
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  char smallbuf[1];
  res = getdomainname (smallbuf, sizeof (domainnamelarge));
  asm volatile ("" : : "r" (res));
  CHK_FAIL_END
#endif

  fd_set s;
  FD_ZERO (&s);

  FD_SET (FD_SETSIZE - 1, &s);
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  FD_SET (FD_SETSIZE, &s);
  CHK_FAIL_END

  CHK_FAIL_START
  FD_SET (l0 + FD_SETSIZE, &s);
  CHK_FAIL_END
#endif

  FD_CLR (FD_SETSIZE - 1, &s);
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  FD_CLR (FD_SETSIZE, &s);
  CHK_FAIL_END

  CHK_FAIL_START
  FD_SET (l0 + FD_SETSIZE, &s);
  CHK_FAIL_END
#endif

  FD_ISSET (FD_SETSIZE - 1, &s);
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  FD_ISSET (FD_SETSIZE, &s);
  CHK_FAIL_END

  CHK_FAIL_START
  FD_ISSET (l0 + FD_SETSIZE, &s);
  CHK_FAIL_END
#endif

  struct pollfd fds[1];
  fds[0].fd = STDOUT_FILENO;
  fds[0].events = POLLOUT;
  poll (fds, 1, 0);
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  poll (fds, 2, 0);
  CHK_FAIL_END

  CHK_FAIL_START
  poll (fds, l0 + 2, 0);
  CHK_FAIL_END
#endif
  ppoll (fds, 1, NULL, NULL);
#if __USE_FORTIFY_LEVEL >= 1
  CHK_FAIL_START
  ppoll (fds, 2, NULL, NULL);
  CHK_FAIL_END

  CHK_FAIL_START
  ppoll (fds, l0 + 2, NULL, NULL);
  CHK_FAIL_END
#endif

  return ret;
}
