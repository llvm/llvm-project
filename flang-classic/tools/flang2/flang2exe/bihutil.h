/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef BIHUTIL_H_
#define BIHUTIL_H_

#include "gbldefs.h"
#include <stdio.h>

/**
   \brief ...
 */
bool any_asm(void);

/**
   \brief ...
 */
int addbih(int after);

/**
   \brief ...
 */
int addnewbih(int after, int flags, int fih);

/**
   \brief ...
 */
int exp_addbih(int after);

/**
   \brief ...
 */
int merge_bih(int curbih);

/**
   \brief ...
 */
void bih_cleanup(void);

/**
   \brief ...
 */
void bih_init(void);

/**
   \brief ...
 */
void *ccff_bih_info(int msgtype, const char *msgid, int bihx, const char *message,
                    ...);

/**
   \brief ...
 */
void delbih(int bihx);

/**
   \brief ...
 */
void dump_blocks(FILE *ff, int bih, const char *fmt, int fihflag);

/**
   \brief ...
 */
void dump_one_block(FILE *ff, int bih, const char *fmt);

/**
   \brief ...
 */
void merge_bih_flags(int to_bih, int fm_bih);

/**
   \brief ...
 */
void merge_blks(int b1, int b2);

/**
   \brief ...
 */
void merge_rgset(int tobih, int frombih, bool reuse_to);

/**
   \brief ...
 */
void split_extended(void);

/**
   \brief ...
 */
void *subccff_bih_info(void *xparent, int msgtype, const char *msgid, int bihx,
                       const char *message, ...);

/**
   \brief ...
 */
void unsplit(void);

/**
   \brief ...
*/
bool contains_par_blocks(void);

#endif
