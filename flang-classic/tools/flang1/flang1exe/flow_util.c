/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *
 * \brief Utility functions for optimizer submodule responsible for performing
 * flow analysis. Used by the optimizer and vectorizer in the frontend and
 * backend.
 */

#include "flow_util.h"

/*
 * Hash of new uses. We were previously using a linear search of opt.useb each
 * time a new use was added for O(N^2), which was impacting compile times.
 * Replacing the linear search with a hash lookup results in O(N).
 */
static hashmap_t use_hash = NULL;

/*
 * Allocate a new use_hash.
 */
void
use_hash_alloc()
{
  use_hash = hashmap_alloc(hash_functions_direct);
} /* use_hash_alloc */

/*
 * Clear and free the use_hash.
 */
void
use_hash_free()
{
  if (use_hash) {
    hashmap_clear(use_hash);
    hashmap_free(use_hash);
    use_hash = NULL;
  }
} /* use_hash_free */

/*
 * Clear, free, and reallocate the use_hash.
 */
void
use_hash_reset()
{
  if (use_hash) {
    use_hash_free();
  }
  use_hash_alloc();
} /* use_hash_reset */

/*
 * Turn the fields of the use into a hash_key_t. Hash function requires to
 * avoid clustering in lower bits of hash.
 *
 * high: In the backend is ilix, frontend is addrx
 * mid: In the backend is nmex, frontend is nmex
 * low: In the backend is iltx, frontend is stdx
 */
static hash_key_t
use_hash_key(bool high_bit, bool use_high, int high, int mid, int low)
{
  long long key;
  // lower USE_HASH_BITS bits of mask set, upper bits cleared.
  long long mask = ((long long)1 << USE_HASH_BITS)-1;

  // This only works if the number of nme, ili, and ilt are each less than 2^USE_HASH_BITS
  if (use_high) {
    assert(high < (int)mask, "too many ILIs to hash", high, ERR_Fatal);
  }
  assert(mid < (int)mask, "too many NMEs to hash", mid, ERR_Fatal);
  assert(low < (int)mask, "too many ILTs to hash", low, ERR_Fatal);

  key = 0;
  if (high_bit)
    key |= (long long)1; // top bit high_bit flag
  key <<= 1;
  if (use_high && !high_bit)
    key |= ((long long)high & mask);
  key <<= USE_HASH_BITS;
  key |= (mid & mask);
  key <<= USE_HASH_BITS;
  key |= (low & mask);

  assert((void*)key != NULL, "NULL key computed", key, ERR_Fatal); 
  assert((void*)key != (void*)~0UL, "~0UL key computed", key, ERR_Fatal); 

  return (hash_key_t)key;
} /* use_hash_key */

/*
 * From the fields of a new use, create a hash key, and insert the use into the
 * use_hash with the key.
 */
void
use_hash_insert(int usex, bool high_bit, bool use_match_ili, int ilix, int nmex, int iltx)
{
  hash_key_t key = use_hash_key(high_bit, use_match_ili, ilix, nmex, iltx);
  hash_data_t data = INT2HKEY(usex);
  hashmap_replace(use_hash, key, &data);
} /* use_hash_insert */

/*
 * From the fields of a potential new use, create a hash key, and check the
 * use_hash to see of that potential use has already been inserted. Return the
 * existing use, if it exists.
 *
 * We were previously using a linear search of opt.useb each time a new use was
 * added for O(N^2), which was impacting compile times.  Replacing the linear
 * search with a hash lookup results in O(N).
 */
int
use_hash_lookup(bool high_bit, bool use_match_ili, int ilix, int nmex, int iltx)
{
  hash_data_t usex = NULL;
  hash_key_t key = use_hash_key(high_bit, use_match_ili, ilix, nmex, iltx);
  if (hashmap_lookup(use_hash, key, &usex)) {
    return HKEY2INT(usex);
  }
  return 0;
} /* use_hash_lookup */

#if DEBUG
/*
 * Helper function to use_hash_dump() to dump a single use_hash entry.
 */
static void
use_hash_dump_one(hash_key_t key, hash_data_t data, void *context)
{
  FILE* dfile = stderr;
  fprintf(dfile, "use_hash key:%zu data:%zu\n", (size_t)key, (size_t)data);
  //_dumpuse(HKEY2INT(data), true);
  fprintf(dfile, "\n");
} /* use_hash_dump_one */

/*
 * Dump routine for the use_hash designed to be called from a debugger.
 */
void
use_hash_dump()
{
  hashmap_iterate(use_hash, use_hash_dump_one, NULL);
} /* use_hash_dump */

/*
 * use_hash utility function to compare the results of the hash search (usex)
 * to the results of the older and slower linear search (lusex).
 */
void
use_hash_check(int usex, int lusex, bool found)
{
  if (found && (usex == lusex)) {
    // Success. Both methods found the same use.
  } else if (found && !usex) {
    // Failure. Linear search found a use, but the hash lookup found nothing.
    assert(0, "use_hash failed should be found", lusex, ERR_Fatal);
  } else if (!found && usex) {
    // Failure. Linear search found nothing, but hash lookup found a use.
    assert(0, "use_hash failed shouldn't be found", usex, ERR_Fatal);
  } else {
    // Success. Boh methods found no use.
  }
} /* use_hash_check */
#endif /* DEBUG */
