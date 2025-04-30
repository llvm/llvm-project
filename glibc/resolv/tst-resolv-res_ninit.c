/* Test the creation of many struct __res_state objects.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <mcheck.h>
#include <resolv.h>
#include <resolv/resolv_context.h>
#include <stdlib.h>
#include <support/check.h>

/* Order the resolver states by their extended resolver state
   index.  */
static int
sort_res_state (const void *a, const void *b)
{
  res_state left = (res_state) a;
  res_state right = (res_state) b;
  return memcmp (&left->_u._ext.__glibc_extension_index,
                 &right->_u._ext.__glibc_extension_index,
                 sizeof (left->_u._ext.__glibc_extension_index));
}

static int
do_test (void)
{
  mtrace ();

  enum { count = 100 * 1000 };
  res_state array = calloc (count, sizeof (*array));
  const struct resolv_conf *conf = NULL;
  for (size_t i = 0; i < count; ++i)
    {
      TEST_VERIFY (res_ninit (array + i) == 0);
      TEST_VERIFY (array[i].nscount > 0);
      struct resolv_context *ctx = __resolv_context_get_override (array + i);
      TEST_VERIFY_EXIT (ctx != NULL);
      TEST_VERIFY (ctx->resp == array + i);
      if (i == 0)
        {
          conf = ctx->conf;
          TEST_VERIFY (conf != NULL);
        }
      else
        /* The underyling configuration should be identical across all
           res_state opjects because resolv.conf did not change.  */
        TEST_VERIFY (ctx->conf == conf);
    }
  qsort (array, count, sizeof (*array), sort_res_state);
  for (size_t i = 1; i < count; ++i)
    /* All extension indices should be different.  */
    TEST_VERIFY (sort_res_state (array + i - 1, array + i) < 0);
  for (size_t i = 0; i < count; ++i)
    res_nclose (array + i);
  free (array);

  TEST_VERIFY (res_init () == 0);
  return 0;
}

#define TIMEOUT 50
#include <support/test-driver.c>
