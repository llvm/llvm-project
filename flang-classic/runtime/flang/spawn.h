/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* host data structure */

struct hostd {
  char name[256]; /* name */
  int pwr;        /* current power */
  int opwr;       /* original power */
  float lav;      /* load average */
  int cnt;        /* number of processes for host */
  int lcpu;       /* lowest logical process on node */
};
