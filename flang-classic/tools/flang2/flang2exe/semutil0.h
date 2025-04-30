/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef SEMUTIL0_H_
#define SEMUTIL0_H_

#include "gbldefs.h"
#include "symtab.h"

#define AREA_SIZE 16

/**
   \brief ...
 */
bool sem_eq_str(int con, const char *pattern);

/**
   \brief ...
 */
INT cngcon(INT oldval, DTYPE oldtyp, DTYPE newtyp);

/**
   \brief ...
 */
int getrval(int ilmptr);

/**
   \brief ...
 */
void semant_init(void);

/**
   \brief ...
 */
void semant_reinit(void);

#endif // SEMUTIL0_H_
