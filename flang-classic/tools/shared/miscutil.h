/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef MISCUTIL_H_
#define MISCUTIL_H_

#include "gbldefs.h"
#include "sharedefs.h"
#include <stdio.h>

/**
   \brief ...
 */
bool is_xflag_bit(int indx);

/**
   \brief ...
 */
char *literal_string(char *oldstr, int userlen, bool isStringW);

/**
   \brief ...
 */
char *mkfname(const char *oldname, const char *oldsuf, const char *newsuf);

/**
   \brief ...
 */
int license_prc2(void);

/**
   \brief ...
 */
int license_prc(void);

/**
   \brief ...
 */
int stg_next_freelist(STG *stg);

/**
   \brief ...
 */
int stg_next(STG *stg, int n);

/**
   \brief ...
 */
void fprintf_str_esc_backslash(FILE *f, char *str);

/**
   \brief ...
 */
void set_xflag(int indx, INT val);

/**
   \brief ...
 */
void set_yflag(int indx, INT val);

/**
   \brief ...
 */
void stg_add_freelist(STG *stg, int r);

/**
   \brief ...
 */
void stg_alloc_sidecar(STG *basestg, STG *stg, int dtsize, const char *name);

/**
   \brief ...
 */
void stg_alloc(STG *stg, int dtsize, int size, const char *name);

/**
   \brief ...
 */
void stg_clear_all(STG *stg);

/**
   \brief ...
 */
void stg_clear_force(STG *stg, BIGUINT64 r, BIGUINT64 n, bool force);

/**
   \brief ...
 */
void stg_clear(STG *stg, int r, int n);

/**
   \brief ...
 */
void stg_delete_sidecar(STG *basestg, STG *stg);

/**
   \brief ...
 */
void stg_delete(STG *stg);

/**
   \brief ...
 */
void stg_need(STG *stg);

/**
   \brief ...
 */
void stg_set_freelink(STG *stg, int offset);

#endif // MISCUTIL_H_
