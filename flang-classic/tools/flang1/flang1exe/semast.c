/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 * semast.c  -  semantic analyzer ast utility routines
 */

#include "gbldefs.h"
#include "global.h"
#include "gramsm.h"
#include "gramtk.h"
#include "symtab.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "ast.h"

/* external references: */

/* contents of this file:  */

/*****************************************************************************/
