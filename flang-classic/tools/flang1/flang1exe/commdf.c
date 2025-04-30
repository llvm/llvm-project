/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Data definitions for communication data structures.
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "gramtk.h"
#include "comm.h"
#include "symutl.h"
#include "extern.h"

TRANSFORM trans = {{NULL, 0, 0},
                   {NULL, 0, 0},
                   {NULL, 0, 0},
                   0,
                   0,
                   0,
                   NULL,
                   NULL,
                   0,
                   0,
                   0,
                   0,
                   0};
struct arg_gbl arg_gbl = {0, 0, FALSE, FALSE};
struct forall_gbl forall_gbl = {0, 0, 0, 0, 0, 0};
struct pre_loop pre_loop = {0, 0, 0, 0};
struct comminfo comminfo = {0, 0, 0, 0, 0, 0, 0, 0, 0, {0, 0, 0, 0, 0, 0, 0}, 0};
struct tbl tbl = {NULL, 0, 0};
struct tbl pertbl = {NULL, 0, 0};
struct tbl gstbl = {NULL, 0, 0};
struct tbl brtbl = {NULL, 0, 0};
