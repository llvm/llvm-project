/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Get distribution for temps.
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "symutl.h"
#include "transfrm.h"
#include "gramtk.h"
#include "extern.h"
