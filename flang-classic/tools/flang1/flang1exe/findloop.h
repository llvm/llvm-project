/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef FINDLOOP_H_
#define FINDLOOP_H_

#include "universal.h"
#include <stdio.h>

/**
   \brief ...
 */
bool contains_loop(int lp1, int lp2);

/**
   \brief ...
 */
bool is_childloop(int lp1, int lp2);

/**
   \brief ...
 */
bool is_dominator(int v, int w);

/**
   \brief ...
 */
bool is_post_dominator(int v, int w);

/**
   \brief ...
 */
bool is_tail_aexe(int lp);

/**
   \brief ...
 */
bool overlapping_loops(int lp1, int lp2);

/**
   \brief ...
 */
void __dump_loop(FILE *ff, int lp);

/**
   \brief ...
 */
void dump_loops(void);

/**
   \brief ...
 */
void __dump_region(FILE *ff, int lp);

/**
   \brief ...
 */
void dump_region(int lp);

/**
   \brief ...
 */
void findloop(int hlopt_bv);

/**
   \brief ...
 */
void findlooptopsort(void);

/**
   \brief ...
 */
void reorderloops(void);

/**
   \brief ...
 */
void sortloops(void);

#endif // FINDLOOP_H_
