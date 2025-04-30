/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef EXPATOMICS_H_
#define EXPATOMICS_H_

#include "gbldefs.h"
#include "symtab.h"
#include "expand.h"
#include "ili.h"

/**
   \brief ...
 */
bool exp_end_atomic(int store, int curilm);

/**
   \brief ...
 */
ILI_OP get_atomic_update_opcode(int current_ili);

/**
   \brief ...
 */
int create_atomic_capture_seq(int update_ili, int read_ili, int capture_first);

/**
   \brief ...
 */
int create_atomic_read_seq(int store_ili);

/**
   \brief ...
 */
int create_atomic_seq(int store_ili);

/**
   \brief ...
 */
int create_atomic_write_seq(int store_ili);

/**
   \brief ...
 */
int exp_mp_atomic_read(ILM *ilmp);

/**
   \brief ...
 */
int get_atomic_capture_created(void);

/**
   \brief ...
 */
int get_atomic_function_ex(ILI_OP opcode);

/**
   \brief ...
 */
int get_atomic_function(ILI_OP opcode);

/**
   \brief ...
 */
int get_atomic_read_opcode(int current_ili);

/**
   \brief ...
 */
int get_atomic_store_created(void);

/**
   \brief ...
 */
int get_atomic_write_opcode(int current_ili);

/**
   \brief ...
 */
int get_capture_read_ili(void);

/**
   \brief ...
 */
int get_capture_update_ili(void);

/**
   \brief ...
 */
int get_is_in_atomic_capture(void);

/**
   \brief ...
 */
int get_is_in_atomic_read(void);

/**
   \brief ...
 */
int get_is_in_atomic(void);

/**
   \brief ...
 */
int get_is_in_atomic_write(void);

/**
   \brief ...
 */
void exp_mp_atomic_capture(ILM *ilmp);

/**
   \brief ...
 */
void exp_mp_atomic_update(ILM *ilmp);

/**
   \brief ...
 */
void exp_mp_atomic_write(ILM *ilmp);

/**
   \brief ...
 */
void set_atomic_capture_created(int x);

/**
   \brief ...
 */
void set_atomic_store_created(int x);

/**
   \brief ...
 */
void set_capture_read_ili(int x);

/**
   \brief ...
 */
void set_capture_update_ili(int x);

/**
   \brief ...
 */
void set_is_in_atomic_capture(int x);

/**
   \brief ...
 */
void set_is_in_atomic(int x);

/**
   \brief ...
 */
void set_is_in_atomic_read(int x);

/**
   \brief ...
 */
void set_is_in_atomic_write(int x);

#endif
