/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdbool.h>

/** \file
 * \brief Definitions for x-flags handling routines
 *
 * X-flags are an array of integers, some of which are bitvectors and some are
 * just plain values that are used to control compiler's behavior. XBIT macros
 * scattered around the source read those values.
 */

/** \brief Query whether x flag index corresponds to a bit vector
 *
 * \return      true is it is, false if it corresponds to a plain value
 * \param index index into x flag array
 */
bool is_xflag_bitvector(int index);

/** \brief Set x flag value
 *
 * XOR passed value if with existing one if the index corresponds to a
 * bitvector, otherwise just assign it.
 *
 * \param xflags pointer to x flags array
 * \param index  index of the element to modify
 * \param value  mask to apply (or value to set)
 */
void set_xflag_value(int *xflags, int index, int value);

/** \brief Unset x flag value
 *
 * Bitwise AND passed value if with existing one if the index corresponds to a
 * bitvector, otherwise just set it to zero.
 *
 * \param xflags pointer to x flags array
 * \param index  index of the element to modify
 * \param value  mask to unset (if value is a bit vector)
 */
void unset_xflag_value(int *xflags, int index, int value);
