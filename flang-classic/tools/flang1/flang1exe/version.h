/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

typedef struct {
  const char *lang;    /* language */
  const char *host;    /* host */
  char *vsn;     /* version number */
  char *bld;     /* build number */
  const char *dvsn;    /* date-based version number */
  const char *target;  /* target compiler */
  const char *product; /* product designation */
  const char *copyright;
} VERSION;

extern VERSION version;

const char *get_version_string(void);

