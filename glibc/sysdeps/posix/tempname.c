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

#if !_LIBC
# include <libc-config.h>
# include "tempname.h"
#endif

#include <sys/types.h>
#include <assert.h>
#include <stdbool.h>

#include <errno.h>

#include <stdio.h>
#ifndef P_tmpdir
# define P_tmpdir "/tmp"
#endif
#ifndef TMP_MAX
# define TMP_MAX 238328
#endif
#ifndef __GT_FILE
# define __GT_FILE      0
# define __GT_DIR       1
# define __GT_NOCREATE  2
#endif
#if !_LIBC && (GT_FILE != __GT_FILE || GT_DIR != __GT_DIR       \
               || GT_NOCREATE != __GT_NOCREATE)
# error report this to bug-gnulib@gnu.org
#endif

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <stdalign.h>
#include <stdint.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>

#if _LIBC
# define struct_stat64 struct __stat64_t64
# define __secure_getenv __libc_secure_getenv
#else
# define struct_stat64 struct stat
# define __gen_tempname gen_tempname
# define __mkdir mkdir
# define __open open
# define __lstat64_time64(file, buf) lstat (file, buf)
# define __stat64(file, buf) stat (file, buf)
# define __getrandom getrandom
# define __clock_gettime64 clock_gettime
# define __timespec64 timespec
#endif

/* Use getrandom if it works, falling back on a 64-bit linear
   congruential generator that starts with Var's value
   mixed in with a clock's low-order bits if available.  */
typedef uint_fast64_t random_value;
#define RANDOM_VALUE_MAX UINT_FAST64_MAX
#define BASE_62_DIGITS 10 /* 62**10 < UINT_FAST64_MAX */
#define BASE_62_POWER (62LL * 62 * 62 * 62 * 62 * 62 * 62 * 62 * 62 * 62)

static random_value
random_bits (random_value var, bool use_getrandom)
{
  random_value r;
  /* Without GRND_NONBLOCK it can be blocked for minutes on some systems.  */
  if (use_getrandom && __getrandom (&r, sizeof r, GRND_NONBLOCK) == sizeof r)
    return r;
#if _LIBC || (defined CLOCK_MONOTONIC && HAVE_CLOCK_GETTIME)
  /* Add entropy if getrandom did not work.  */
  struct __timespec64 tv;
  __clock_gettime64 (CLOCK_MONOTONIC, &tv);
  var ^= tv.tv_nsec;
#endif
  return 2862933555777941757 * var + 3037000493;
}

#if _LIBC
/* Return nonzero if DIR is an existent directory.  */
static int
direxists (const char *dir)
{
  struct_stat64 buf;
  return __stat64_time64 (dir, &buf) == 0 && S_ISDIR (buf.st_mode);
}

/* Path search algorithm, for tmpnam, tmpfile, etc.  If DIR is
   non-null and exists, uses it; otherwise uses the first of $TMPDIR,
   P_tmpdir, /tmp that exists.  Copies into TMPL a template suitable
   for use with mk[s]temp.  Will fail (-1) if DIR is non-null and
   doesn't exist, none of the searched dirs exists, or there's not
   enough space in TMPL. */
int
__path_search (char *tmpl, size_t tmpl_len, const char *dir, const char *pfx,
               int try_tmpdir)
{
  const char *d;
  size_t dlen, plen;

  if (!pfx || !pfx[0])
    {
      pfx = "file";
      plen = 4;
    }
  else
    {
      plen = strlen (pfx);
      if (plen > 5)
        plen = 5;
    }

  if (try_tmpdir)
    {
      d = __secure_getenv ("TMPDIR");
      if (d != NULL && direxists (d))
        dir = d;
      else if (dir != NULL && direxists (dir))
        /* nothing */ ;
      else
        dir = NULL;
    }
  if (dir == NULL)
    {
      if (direxists (P_tmpdir))
        dir = P_tmpdir;
      else if (strcmp (P_tmpdir, "/tmp") != 0 && direxists ("/tmp"))
        dir = "/tmp";
      else
        {
          __set_errno (ENOENT);
          return -1;
        }
    }

  dlen = strlen (dir);
  while (dlen > 1 && dir[dlen - 1] == '/')
    dlen--;                     /* remove trailing slashes */

  /* check we have room for "${dir}/${pfx}XXXXXX\0" */
  if (tmpl_len < dlen + 1 + plen + 6 + 1)
    {
      __set_errno (EINVAL);
      return -1;
    }

  sprintf (tmpl, "%.*s/%.*sXXXXXX", (int) dlen, dir, (int) plen, pfx);
  return 0;
}
#endif /* _LIBC */

#if _LIBC
static int try_tempname_len (char *, int, void *, int (*) (char *, void *),
                             size_t);
#endif

static int
try_file (char *tmpl, void *flags)
{
  int *openflags = flags;
  return __open (tmpl,
                 (*openflags & ~O_ACCMODE)
                 | O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
}

static int
try_dir (char *tmpl, void *flags _GL_UNUSED)
{
  return __mkdir (tmpl, S_IRUSR | S_IWUSR | S_IXUSR);
}

static int
try_nocreate (char *tmpl, void *flags _GL_UNUSED)
{
  struct_stat64 st;

  if (__lstat64_time64 (tmpl, &st) == 0 || errno == EOVERFLOW)
    __set_errno (EEXIST);
  return errno == ENOENT ? 0 : -1;
}

/* These are the characters used in temporary file names.  */
static const char letters[] =
"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

/* Generate a temporary file name based on TMPL.  TMPL must match the
   rules for mk[s]temp (i.e., end in at least X_SUFFIX_LEN "X"s,
   possibly with a suffix).
   The name constructed does not exist at the time of the call to
   this function.  TMPL is overwritten with the result.

   KIND may be one of:
   __GT_NOCREATE:       simply verify that the name does not exist
                        at the time of the call.
   __GT_FILE:           create the file using open(O_CREAT|O_EXCL)
                        and return a read-write fd.  The file is mode 0600.
   __GT_DIR:            create a directory, which will be mode 0700.

   We use a clever algorithm to get hard-to-predict names. */
#ifdef _LIBC
static
#endif
int
gen_tempname_len (char *tmpl, int suffixlen, int flags, int kind,
                  size_t x_suffix_len)
{
  static int (*const tryfunc[]) (char *, void *) =
    {
      [__GT_FILE] = try_file,
      [__GT_DIR] = try_dir,
      [__GT_NOCREATE] = try_nocreate
    };
  return try_tempname_len (tmpl, suffixlen, &flags, tryfunc[kind],
                           x_suffix_len);
}

#ifdef _LIBC
static
#endif
int
try_tempname_len (char *tmpl, int suffixlen, void *args,
                  int (*tryfunc) (char *, void *), size_t x_suffix_len)
{
  size_t len;
  char *XXXXXX;
  unsigned int count;
  int fd = -1;
  int save_errno = errno;

  /* A lower bound on the number of temporary files to attempt to
     generate.  The maximum total number of temporary file names that
     can exist for a given template is 62**6.  It should never be
     necessary to try all of these combinations.  Instead if a reasonable
     number of names is tried (we define reasonable as 62**3) fail to
     give the system administrator the chance to remove the problems.
     This value requires that X_SUFFIX_LEN be at least 3.  */
#define ATTEMPTS_MIN (62 * 62 * 62)

  /* The number of times to attempt to generate a temporary file.  To
     conform to POSIX, this must be no smaller than TMP_MAX.  */
#if ATTEMPTS_MIN < TMP_MAX
  unsigned int attempts = TMP_MAX;
#else
  unsigned int attempts = ATTEMPTS_MIN;
#endif

  /* A random variable.  The initial value is used only the for fallback path
     on 'random_bits' on 'getrandom' failure.  Its initial value tries to use
     some entropy from the ASLR and ignore possible bits from the stack
     alignment.  */
  random_value v = ((uintptr_t) &v) / alignof (max_align_t);

  /* How many random base-62 digits can currently be extracted from V.  */
  int vdigits = 0;

  /* Whether to consume entropy when acquiring random bits.  On the
     first try it's worth the entropy cost with __GT_NOCREATE, which
     is inherently insecure and can use the entropy to make it a bit
     less secure.  On the (rare) second and later attempts it might
     help against DoS attacks.  */
  bool use_getrandom = tryfunc == try_nocreate;

  /* Least unfair value for V.  If V is less than this, V can generate
     BASE_62_DIGITS digits fairly.  Otherwise it might be biased.  */
  random_value const unfair_min
    = RANDOM_VALUE_MAX - RANDOM_VALUE_MAX % BASE_62_POWER;

  len = strlen (tmpl);
  if (len < x_suffix_len + suffixlen
      || strspn (&tmpl[len - x_suffix_len - suffixlen], "X") < x_suffix_len)
    {
      __set_errno (EINVAL);
      return -1;
    }

  /* This is where the Xs start.  */
  XXXXXX = &tmpl[len - x_suffix_len - suffixlen];

  for (count = 0; count < attempts; ++count)
    {
      for (size_t i = 0; i < x_suffix_len; i++)
        {
          if (vdigits == 0)
            {
              do
                {
                  v = random_bits (v, use_getrandom);
                  use_getrandom = true;
                }
              while (unfair_min <= v);

              vdigits = BASE_62_DIGITS;
            }

          XXXXXX[i] = letters[v % 62];
          v /= 62;
          vdigits--;
        }

      fd = tryfunc (tmpl, args);
      if (fd >= 0)
        {
          __set_errno (save_errno);
          return fd;
        }
      else if (errno != EEXIST)
        return -1;
    }

  /* We got out of the loop because we ran out of combinations to try.  */
  __set_errno (EEXIST);
  return -1;
}

int
__gen_tempname (char *tmpl, int suffixlen, int flags, int kind)
{
  return gen_tempname_len (tmpl, suffixlen, flags, kind, 6);
}

#if !_LIBC
int
try_tempname (char *tmpl, int suffixlen, void *args,
              int (*tryfunc) (char *, void *))
{
  return try_tempname_len (tmpl, suffixlen, args, tryfunc, 6);
}
#endif
