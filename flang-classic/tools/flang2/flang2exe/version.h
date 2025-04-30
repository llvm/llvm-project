/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef VERSION_H_
#define VERSION_H_

typedef struct {
  const char *lang;    /* language */
  const char *host;    /* host */
  const char *vsn;     /* version number */
  const char *bld;     /* build number */
  const char *dvsn;    /* date-based version number */
  const char *target;  /* target compiler */
  const char *product; /* product designation */
  const char *copyright;
} VERSION;

extern VERSION version;

/// \brief Get a string composed of version and build numbers
const char *get_version_string(void);


#endif // VERSION_H_
