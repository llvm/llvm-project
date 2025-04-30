/* Duplicate a response context used in DNS resolver tests.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <support/resolv_test.h>
#include <support/support.h>

struct resolv_response_context *
resolv_response_context_duplicate (const struct resolv_response_context *ctx)
{
  struct resolv_response_context *result = xmalloc (sizeof (*result));
  memcpy (result, ctx, sizeof (*result));
  if (result->client_address != NULL)
    {
      result->client_address = xmalloc (result->client_address_length);
      memcpy (result->client_address, ctx->client_address,
              result->client_address_length);
    }
  result->query_buffer = xmalloc (result->query_length);
  memcpy (result->query_buffer, ctx->query_buffer, result->query_length);
  return result;
}
