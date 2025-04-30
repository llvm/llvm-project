/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Empty semantic analyzer routines which process executable
        statements (not I/O & not HPF statements).
 */

#include "gbldefs.h"
#include "gramsm.h"
#include "gramtk.h"
#include "global.h"
#include "symtab.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "ast.h"
#include "dinit.h"

/** \brief Semantic actions - part 3.
    \param rednum reduction number
    \param top    top of stack after reduction
 */
void
psemant3(int rednum, SST *top)
{

  SST_ASTP(LHS, 0);
}
