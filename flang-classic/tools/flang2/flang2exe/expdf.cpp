/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "ilm.h"
/* need ilmtp.h since expand.h tests #ifdef IM_... */
#include "ilmtp.h"
#include "ili.h"
#define EXPANDER_DECLARE_INTERNAL
#include "expand.h"
#include "regutil.h"

ILIB ilib;

ILTB iltb;

BIHB bihb;

NMEB nmeb;

EXP expb;

RCANDB rcandb;

RATB ratb;
