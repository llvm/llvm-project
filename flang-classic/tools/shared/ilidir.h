/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ILIDIR_H_
#define ILIDIR_H_

/**
   \brief ...
 */
void close_pragma(void);

/**
   \brief ...
 */
void ili_lpprg_init(void);

/**
   \brief ...
 */
void open_pragma(int line);

/**
   \brief ...
 */
void pop_pragma(void);

/**
   \brief ...
 */
void push_pragma(int line);

#endif // ILIDIR_H_
