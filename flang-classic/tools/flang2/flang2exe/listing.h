/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LISTING_H_
#define LISTING_H_

#include <stdio.h>

/**
   \brief ...
 */
void list_init(FILE *fd);

/**
   \brief ...
 */
void list_line(const char *txt);

/**
   \brief ...
 */
void list_page(void);

#endif // LISTING_H_
