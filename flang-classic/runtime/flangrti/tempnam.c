/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if !defined(_WIN64) && !defined(USETEMPNAM) /* { */
#include <errno.h>
#include <stdio.h>

/* This is really defined in stdio.h.  But we don't use the real stdio.h */

#define P_tmpdir "/tmp"

extern unsigned long strlen(const char *);
/* FIXME: #include <stdlib.h> , may have to keep externs for Windows */
extern void *malloc(unsigned long);
extern char *getenv(const char *);

static int rand;

/* chars for base 32 and 64 conversions */

static char chars[] = {
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789XW"};

/* add a string to the temp name */

static char *
add(char *p, const char *q, int n)
{
  while ((*q != '\0') && (n-- != 0)) {
    *p++ = *q++;
  }
  *p = '\0';
  return (p);
}

/* add a number to the temp name, first char is base 32, the rest is base 64 */

static char *
addn(char *p, int val, int n)
{
  char bufn[9] = {0}; /* avoid some valgrind uninitialized errors */
  int d;
  int i;

  i = 0;
  while (i < n) {
    d = val & 0x3f;
    val >>= 6;
    bufn[i] = chars[d];
    ++i;
    if (val == 0)
      break;
  }
  p = add(p, bufn, i);
  return (p);
}

/* generate a possible temp name */

static char *
gentmp(const char *dir, const char *pfx)
{
  extern long getpid(void);
  extern long time(void *);
  char *buf;
  char *p, *q;
  const char *tmp;
  int n;

  tmp = getenv("TMPDIR");
  if (tmp == NULL || tmp[0] == '\0') {
    tmp = getenv("TMP");
    if (tmp == NULL || tmp[0] == '\0') {
      tmp = dir;
      if (tmp == NULL || tmp[0] == '\0') {
        tmp = P_tmpdir;
      }
    }
  }
  buf = p = malloc(strlen(tmp) + 32);
  if (p == NULL) {
    return (NULL);
  }
  p = add(p, tmp, -1);
  p = add(p, "/", -1);
  if (pfx != NULL) {
    p = add(p, pfx, 5);
  }

  if (rand == 0) { /* first time, create seed */
    rand = 0;
    n = 0;
    q = getenv("USER");
    q = (q == NULL ? getenv("USERNAME") : q);
    if (q != NULL) {
      while (*q != '\0') {
        rand ^= (*q++) << n++;
      }
    }
    n = 0;
    q = getenv("HOSTNAME");
    if (q != NULL) {
      while (*q != '\0') {
        rand ^= (*q++) << n++;
      }
    }
    rand ^= getpid();
    rand ^= (int)((long)buf) >> 4;
    rand ^= time((long *)0);
  }

  *p++ = chars[rand & 0x1f];
  rand = (rand << 16) + rand * 3;
  p = addn(p, rand, 4);
  rand = (rand << 16) + rand * 3;
  p = addn(p, rand, 4);
  rand = (rand << 16) + rand * 3;
  p = addn(p, rand, 4);
  return (buf);
}

/* generate a temp name that doesn't exist at the moment */

char *
__io_tempnam(const char *dir, const char *pfx)
{
  extern int access(const char *, int);
  char *p;

  while (1) {
    p = gentmp(dir, pfx);
    if ((access(p, 0) == -1) && (errno == ENOENT)) {
      break;
    }
  }
  return (p);
}
#else /* }else{ */

extern char *tempnam(char *, char *);

char *
__io_tempnam(char *dir, char *pfx)
{
#if defined(_WIN64)
  return (_tempnam(dir, pfx));
#else
  return (tempnam(dir, pfx));
#endif
}

#endif /* } */
