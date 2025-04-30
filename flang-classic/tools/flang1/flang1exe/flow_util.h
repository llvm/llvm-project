/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef FLOW_UTIL_H_
#define FLOW_UTIL_H_

#include "gbldefs.h"
#include "global.h"
#include "hash.h"
#include "error.h"

/* Turn on or off (when 0) debugging of the use_hash. */
#define DEBUG_USE_HASH 0

/* Number of bits to use in the use_hash_key for each field */
#define USE_HASH_BITS 21

/*
 * \brief Allocate the use_hash.
 */
void use_hash_alloc();

/*
 * \brief Free and clear the use_hash.
 */
void use_hash_free();

/*
 * \brief Free, clear, and reallocate the use_hash.
 */
void use_hash_reset();

/*
 * \brief Insert a new use into the use_hash.
 */
void use_hash_insert(int usex, bool high_bit, bool use_match_ili, int ilix, int nmex, int iltx);

/*
 * \brief Lookup a use in the use_hash based on its fields. This is used to
 * check to see if a use already exists, so we don't create duplicates.
 */
int use_hash_lookup(bool high_bit, bool use_match_ili, int ilix, int nmex, int iltx);

/*
 * \brief When DEBUG_USE_HASH is enabled, compare results found by the old
 * quadratic search to the use_hash_lookup result.
 */
void use_hash_check(int usex, int lusex, bool found);

#endif /* FLOW_UTIL_H_ */
