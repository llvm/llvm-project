#include <array_length.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <libc-diag.h>

static int
t1 (void)
{
  int n = -1;
  sscanf ("abc  ", "abc %n", &n);
  printf ("t1: count=%d\n", n);

  return n != 5;
}

static int
t2 (void)
{
  int result = 0;
  int n;
  long N;
  int retval;
#define SCAN(INPUT, FORMAT, VAR, EXP_RES, EXP_VAL) \
  VAR = -1; \
  retval = sscanf (INPUT, FORMAT, &VAR); \
  printf ("sscanf (\"%s\", \"%s\", &x) => %d, x = %ld\n", \
	  INPUT, FORMAT, retval, (long int) VAR); \
  result |= retval != EXP_RES || VAR != EXP_VAL

  /* This function is testing corner cases of the scanf format string,
     so they do not all conform to -Wformat's expectations.  */
  DIAG_PUSH_NEEDS_COMMENT;
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat");
  DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wformat-extra-args");

  SCAN ("12345", "%ld", N, 1, 12345);
  SCAN ("12345", "%llllld", N, 0, -1);
  SCAN ("12345", "%LLLLLd", N, 0, -1);
  SCAN ("test ", "%*s%n",  n, 0, 4);
  SCAN ("test ", "%2*s%n",  n, 0, -1);
  SCAN ("12 ",   "%l2d",  n, 0, -1);
  SCAN ("12 ",   "%2ld",  N, 1, 12);

  n = -1;
  N = -1;
  retval = sscanf ("1 1", "%d %Z", &n, &N);
  printf ("sscanf (\"1 1\", \"%%d %%Z\", &n, &N) => %d, n = %d, N = %ld\n", \
	  retval, n, N); \
  result |= retval != 1 || n != 1 || N != -1;

  DIAG_POP_NEEDS_COMMENT;

  return result;
}

static int
t3 (void)
{
  char buf[80];
  wchar_t wbuf[80];
  int result = 0;
  int retval;

  retval = sprintf (buf, "%p", (char *) NULL);
  result |= retval != 5 || strcmp (buf, "(nil)") != 0;

  retval = swprintf (wbuf, array_length (wbuf), L"%p", (char *) NULL);
  result |= retval != 5 || wcscmp (wbuf, L"(nil)") != 0;

  return result;
}

volatile double qnanval;
volatile long double lqnanval;
/* A sNaN is only guaranteed to be representable in variables with static (or
   thread-local) storage duration.  */
static volatile double snanval = __builtin_nans ("");
static volatile double msnanval = -__builtin_nans ("");
static volatile long double lsnanval = __builtin_nansl ("");
static volatile long double lmsnanval = -__builtin_nansl ("");
volatile double infval;
volatile long double linfval;


static int
F (void)
{
  char buf[80];
  wchar_t wbuf[40];
  int result = 0;

  qnanval = NAN;

  /* The %f and %F arguments are in fact constants, but GCC is
     prevented from seeing this (volatile is used) so it cannot tell
     that the output is not truncated.  */
  DIAG_PUSH_NEEDS_COMMENT;
#if __GNUC_PREREQ (7, 0)
  DIAG_IGNORE_NEEDS_COMMENT (7.0, "-Wformat-truncation");
#endif

  snprintf (buf, sizeof buf, "%a %A %e %E %f %F %g %G",
	    qnanval, qnanval, qnanval, qnanval,
	    qnanval, qnanval, qnanval, qnanval);
  result |= strcmp (buf, "nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected \"nan NAN nan NAN nan NAN nan NAN\", got \"%s\"\n", buf);

  snprintf (buf, sizeof buf, "%a %A %e %E %f %F %g %G",
	    -qnanval, -qnanval, -qnanval, -qnanval,
	    -qnanval, -qnanval, -qnanval, -qnanval);
  result |= strcmp (buf, "-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected \"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got \"%s\"\n",
	  buf);

  snprintf (buf, sizeof buf, "%a %A %e %E %f %F %g %G",
	    snanval, snanval, snanval, snanval,
	    snanval, snanval, snanval, snanval);
  result |= strcmp (buf, "nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected \"nan NAN nan NAN nan NAN nan NAN\", got \"%s\"\n", buf);

  snprintf (buf, sizeof buf, "%a %A %e %E %f %F %g %G",
	    msnanval, msnanval, msnanval, msnanval,
	    msnanval, msnanval, msnanval, msnanval);
  result |= strcmp (buf, "-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected \"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got \"%s\"\n",
	  buf);

  infval = DBL_MAX * DBL_MAX;

  snprintf (buf, sizeof buf, "%a %A %e %E %f %F %g %G",
	    infval, infval, infval, infval, infval, infval, infval, infval);
  result |= strcmp (buf, "inf INF inf INF inf INF inf INF") != 0;
  printf ("expected \"inf INF inf INF inf INF inf INF\", got \"%s\"\n", buf);

  snprintf (buf, sizeof buf, "%a %A %e %E %f %F %g %G",
	    -infval, -infval, -infval, -infval,
	    -infval, -infval, -infval, -infval);
  result |= strcmp (buf, "-inf -INF -inf -INF -inf -INF -inf -INF") != 0;
  printf ("expected \"-inf -INF -inf -INF -inf -INF -inf -INF\", got \"%s\"\n",
	  buf);

  swprintf (wbuf, array_length (wbuf), L"%a %A %e %E %f %F %g %G",
	    qnanval, qnanval, qnanval, qnanval,
	    qnanval, qnanval, qnanval, qnanval);
  result |= wcscmp (wbuf, L"nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected L\"nan NAN nan NAN nan NAN nan NAN\", got L\"%S\"\n", wbuf);

  swprintf (wbuf, array_length (wbuf), L"%a %A %e %E %f %F %g %G",
	    -qnanval, -qnanval, -qnanval, -qnanval,
	    -qnanval, -qnanval, -qnanval, -qnanval);
  result |= wcscmp (wbuf, L"-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected L\"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got L\"%S\"\n",
	  wbuf);

  swprintf (wbuf, array_length (wbuf), L"%a %A %e %E %f %F %g %G",
	    snanval, snanval, snanval, snanval,
	    snanval, snanval, snanval, snanval);
  result |= wcscmp (wbuf, L"nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected L\"nan NAN nan NAN nan NAN nan NAN\", got L\"%S\"\n", wbuf);

  swprintf (wbuf, array_length (wbuf), L"%a %A %e %E %f %F %g %G",
	    msnanval, msnanval, msnanval, msnanval,
	    msnanval, msnanval, msnanval, msnanval);
  result |= wcscmp (wbuf, L"-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected L\"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got L\"%S\"\n",
	  wbuf);

  swprintf (wbuf, array_length (wbuf), L"%a %A %e %E %f %F %g %G",
	    infval, infval, infval, infval, infval, infval, infval, infval);
  result |= wcscmp (wbuf, L"inf INF inf INF inf INF inf INF") != 0;
  printf ("expected L\"inf INF inf INF inf INF inf INF\", got L\"%S\"\n", wbuf);

  swprintf (wbuf, array_length (wbuf), L"%a %A %e %E %f %F %g %G",
	    -infval, -infval, -infval, -infval,
	    -infval, -infval, -infval, -infval);
  result |= wcscmp (wbuf, L"-inf -INF -inf -INF -inf -INF -inf -INF") != 0;
  printf ("expected L\"-inf -INF -inf -INF -inf -INF -inf -INF\", got L\"%S\"\n",
	  wbuf);

  lqnanval = NAN;

  snprintf (buf, sizeof buf, "%La %LA %Le %LE %Lf %LF %Lg %LG",
	    lqnanval, lqnanval, lqnanval, lqnanval,
	    lqnanval, lqnanval, lqnanval, lqnanval);
  result |= strcmp (buf, "nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected \"nan NAN nan NAN nan NAN nan NAN\", got \"%s\"\n", buf);

  snprintf (buf, sizeof buf, "%La %LA %Le %LE %Lf %LF %Lg %LG",
	    -lqnanval, -lqnanval, -lqnanval, -lqnanval,
	    -lqnanval, -lqnanval, -lqnanval, -lqnanval);
  result |= strcmp (buf, "-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected \"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got \"%s\"\n",
	  buf);

  snprintf (buf, sizeof buf, "%La %LA %Le %LE %Lf %LF %Lg %LG",
	    lsnanval, lsnanval, lsnanval, lsnanval,
	    lsnanval, lsnanval, lsnanval, lsnanval);
  result |= strcmp (buf, "nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected \"nan NAN nan NAN nan NAN nan NAN\", got \"%s\"\n", buf);

  snprintf (buf, sizeof buf, "%La %LA %Le %LE %Lf %LF %Lg %LG",
	    lmsnanval, lmsnanval, lmsnanval, lmsnanval,
	    lmsnanval, lmsnanval, lmsnanval, lmsnanval);
  result |= strcmp (buf, "-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected \"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got \"%s\"\n",
	  buf);

  linfval = LDBL_MAX * LDBL_MAX;

  snprintf (buf, sizeof buf, "%La %LA %Le %LE %Lf %LF %Lg %LG",
	    linfval, linfval, linfval, linfval,
	    linfval, linfval, linfval, linfval);
  result |= strcmp (buf, "inf INF inf INF inf INF inf INF") != 0;
  printf ("expected \"inf INF inf INF inf INF inf INF\", got \"%s\"\n", buf);

  snprintf (buf, sizeof buf, "%La %LA %Le %LE %Lf %LF %Lg %LG",
	    -linfval, -linfval, -linfval, -linfval,
	    -linfval, -linfval, -linfval, -linfval);
  result |= strcmp (buf, "-inf -INF -inf -INF -inf -INF -inf -INF") != 0;
  printf ("expected \"-inf -INF -inf -INF -inf -INF -inf -INF\", got \"%s\"\n",
	  buf);

  swprintf (wbuf, array_length (wbuf),
	    L"%La %LA %Le %LE %Lf %LF %Lg %LG",
	    lqnanval, lqnanval, lqnanval, lqnanval,
	    lqnanval, lqnanval, lqnanval, lqnanval);
  result |= wcscmp (wbuf, L"nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected L\"nan NAN nan NAN nan NAN nan NAN\", got L\"%S\"\n", wbuf);

  swprintf (wbuf, array_length (wbuf),
	    L"%La %LA %Le %LE %Lf %LF %Lg %LG",
	    -lqnanval, -lqnanval, -lqnanval, -lqnanval,
	    -lqnanval, -lqnanval, -lqnanval, -lqnanval);
  result |= wcscmp (wbuf, L"-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected L\"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got L\"%S\"\n",
	  wbuf);

  swprintf (wbuf, array_length (wbuf),
	    L"%La %LA %Le %LE %Lf %LF %Lg %LG",
	    lsnanval, lsnanval, lsnanval, lsnanval,
	    lsnanval, lsnanval, lsnanval, lsnanval);
  result |= wcscmp (wbuf, L"nan NAN nan NAN nan NAN nan NAN") != 0;
  printf ("expected L\"nan NAN nan NAN nan NAN nan NAN\", got L\"%S\"\n", wbuf);

  swprintf (wbuf, array_length (wbuf),
	    L"%La %LA %Le %LE %Lf %LF %Lg %LG",
	    lmsnanval, lmsnanval, lmsnanval, lmsnanval,
	    lmsnanval, lmsnanval, lmsnanval, lmsnanval);
  result |= wcscmp (wbuf, L"-nan -NAN -nan -NAN -nan -NAN -nan -NAN") != 0;
  printf ("expected L\"-nan -NAN -nan -NAN -nan -NAN -nan -NAN\", got L\"%S\"\n",
	  wbuf);

  swprintf (wbuf, array_length (wbuf),
	    L"%La %LA %Le %LE %Lf %LF %Lg %LG",
	    linfval, linfval, linfval, linfval,
	    linfval, linfval, linfval, linfval);
  result |= wcscmp (wbuf, L"inf INF inf INF inf INF inf INF") != 0;
  printf ("expected L\"inf INF inf INF inf INF inf INF\", got L\"%S\"\n", wbuf);

  swprintf (wbuf, array_length (wbuf),
	    L"%La %LA %Le %LE %Lf %LF %Lg %LG",
	    -linfval, -linfval, -linfval, -linfval,
	    -linfval, -linfval, -linfval, -linfval);
  result |= wcscmp (wbuf, L"-inf -INF -inf -INF -inf -INF -inf -INF") != 0;
  printf ("expected L\"-inf -INF -inf -INF -inf -INF -inf -INF\", got L\"%S\"\n",
	  wbuf);

  DIAG_POP_NEEDS_COMMENT;

  return result;
}

int
main (int argc, char *argv[])
{
  int result = 0;

  result |= t1 ();
  result |= t2 ();
  result |= t3 ();
  result |= F ();

  result |= fflush (stdout) == EOF;

  return result;
}
