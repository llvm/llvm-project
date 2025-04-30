/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef DINITUTL_H_
#define DINITUTL_H_

#include "gbldefs.h"
#include "symtab.h"

struct DREC;

/**
   \brief ...
 */
bool df_is_open(void);

/**
   \brief ...
 */
struct DREC *dinit_read(void);

/**
   \brief ...
 */
long dinit_ftell(void);

/**
   \brief ...
 */
void dinit_end(void);

/**
   \brief ...
 */
void dinit_fseek(long off);

/**
   \brief ...
 */
void dinit_fskip(long off);

/**
   \brief ...
 */
void dinit_init(void);

/**
   \brief ...
 */
void dinit_put(DTYPE dtype, ISZ_T conval);

/**
   \brief ...
 */
void dinit_put_string(ISZ_T len, char *str);

/**
   \brief ...
 */
void dinit_read_string(ISZ_T len, char *str);

/**
   \brief ...
 */
void dinit_restore(void);

/**
   \brief ...
 */
void dinit_save(void);

#endif
