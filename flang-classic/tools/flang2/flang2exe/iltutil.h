/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ILTUTIL_H_
#define ILTUTIL_H_

#include "gbldefs.h"
#include <stdio.h>

/**
   \brief ...
 */
int addilt(int after, int ilix);

/**
   \brief ...
 */
int reduce_ilt(int iltx, int ilix);

/**
   \brief ...
 */
void *ccff_ilt_info(int msgtype, const char *msgid, int iltx, int bihx, const char *message, ...);

/**
   \brief ...
 */
void delilt(int iltx);

/**
   \brief ...
 */
void dmpilt(int bihx);

/**
   \brief ...
 */
void dump_ilt(FILE *ff, int bihx);

/**
   \brief ...
 */
void ilt_cleanup(void);

/**
   \brief ...
 */
void ilt_init(void);

/**
   \brief ...
 */
void moveilt(int iltx, int before);

/**
   \brief ...
 */
void rdilts(int bihx);

/**
   \brief ...
 */
void relnkilt(int iltx, int bihx);

/**
   \brief ...
 */
void *subccff_ilt_info(void *xparent, int msgtype, const char *msgid, int iltx, int bihx, const char *message, ...);

/**
   \brief ...
 */
void unlnkilt(int iltx, int bihx, bool reuse);

/**
   \brief ...
 */
void wrilts(int bihx);

#endif // ILTUTIL_H_
