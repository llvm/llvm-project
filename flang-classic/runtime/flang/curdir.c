/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <sys/types.h>
#if !defined(_WIN64)
#include <sys/param.h>
#include <sys/utsname.h>
#endif
#include <stdlib.h>
#include "stdioInterf.h"
#include "fioMacros.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#if defined(_WIN64)
#define getcwd _getcwd
#endif

WIN_MSVCRT_IMP char *WIN_CDECL getenv(const char *);

/* fix pathname for "funny" NFS mount points */

void __fort_fixmnt(char *new, char *old)
{
  const char *q;
  char s[MAXPATHLEN]; /* substitute patterns */
  char *smat;         /* match string */
  char *srep;         /* replace string */
  char *snxt;         /* next pattern */
  int n;

  q = __fort_getopt("-mount"); /* pattern */
  if (q == NULL) {
    q = "/tmp_mnt";
  }
  strcpy(s, q);

  snxt = s;
  while (snxt != NULL) {
    smat = snxt;
    snxt = strchr(snxt, ',');
    if (snxt != NULL) {
      *snxt = '\0';
      snxt++;
    }
    srep = strchr(smat, ':'); /* replace string */
    if (srep != NULL) {
      *srep = '\0';
      srep++;
    }
    n = strlen(smat); /* match string length */
    if (strncmp(old, smat, n) == 0) {
      strcpy(new, srep ? srep : "");
      strcat(new, old + n);
      return;
    }
  }
  strcpy(new, old);
}

/* get current working directory */

void __fort_getdir(char *curdir)
{
  char path[MAXPATHLEN];
  char *p;

  p = getcwd(path, MAXPATHLEN);
  if (p == NULL) {
    p = getenv("PWD");
    if (p == NULL) {
      __fort_abort("cannot find current directory\n");
    }
    strcpy(path, p);
  }
  __fort_fixmnt(curdir, path);
}

/* get current hostname */

void __fort_gethostname(char *host)
{
#if !defined(_WIN64)
  struct utsname un;
#endif
  const char *p;
  int s;

  p = __fort_getopt("-curhost");
  if (p == NULL) {
#if !defined(_WIN64)
    s = uname(&un); /* get hostname */
    if (s == -1) {
      __fort_abortp("uname");
    }
    p = un.nodename;
#endif
  }
  strcpy(host, p);
}
