/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef XREF_H_
#define XREF_H_

#include "gbldefs.h"
#include "global.h"

/**
   \brief ...
 */
void par_xref_put(int lineno, SPTR sym, int sc);

/**
   \brief ...
 */
void par_xref(void);

/**
   \brief ...
 */
void xrefinit(void);

/**
   \brief ...
 */
void xrefput(SPTR symptr, int usage);

/**
   \brief ...
 */
void xref(void);

#endif // XREF_H_
