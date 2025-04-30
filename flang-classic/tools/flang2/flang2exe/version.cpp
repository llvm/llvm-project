/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "gbldefs.h"
#include "release.h"
#include "version.h"

#ifndef LANGUAGE
#define LANGUAGE "F90"
#endif

#define PRODUCT ""

/* COPYRIGHT is extern to make it easy to find in symbol table */
/* it also has extra space to patch in interesting stuff */
const char COPYRIGHT[128] =
    "";

VERSION version = {LANGUAGE, VHOST, VSN, BLD, DVSN, TARGET, PRODUCT, COPYRIGHT};

const char *
get_version_string(void)
{
  static char buf[128];
  snprintf(buf, 128, "%s%s", version.vsn, version.bld);
  return buf;
}
