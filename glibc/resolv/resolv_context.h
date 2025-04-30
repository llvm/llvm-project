/* Temporary, thread-local resolver state.
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

/* struct resolv_context objects are allocated on the heap,
   initialized by __resolv_context_get (and its variants), and
   destroyed by __resolv_context_put.

   A nested call to __resolv_context_get (after another call to
   __resolv_context_get without a matching __resolv_context_put call,
   on the same thread) returns the original pointer, instead of
   allocating a new context.  This prevents unexpected reloading of
   the resolver configuration.  Care is taken to keep the context in
   sync with the thread-local _res object.  (This does not happen with
   __resolv_context_get_override.)

   In contrast to struct __res_state, struct resolv_context is not
   affected by ABI compatibility concerns.

   For the benefit of the res_n* functions, a struct __res_state
   pointer is included in the context object, and a separate
   initialization function is provided.  */

#ifndef _RESOLV_CONTEXT_H
#define _RESOLV_CONTEXT_H

#include <bits/types/res_state.h>
#include <resolv/resolv_conf.h>
#include <stdbool.h>
#include <stddef.h>

/* Temporary resolver state.  */
struct resolv_context
{
  struct __res_state *resp;     /* Backing resolver state.   */

  /* Extended resolver state.  This is set to NULL if the
     __resolv_context_get functions are unable to locate an associated
     extended state.  In this case, the configuration data in *resp
     has to be used; otherwise, the data from *conf should be
     preferred (because it is a superset).  */
  struct resolv_conf *conf;

  /* The following fields are for internal use within the
     resolv_context module.  */
  size_t __refcount;            /* Count of reusages by the get functions.  */
  bool __from_res;              /* True if created from _res.  */

  /* Single-linked list of resolver contexts.  Used for memory
     deallocation on thread cancellation.  */
  struct resolv_context *__next;
};

/* Return the current temporary resolver context, or NULL if there was
   an error (indicated by errno).  A call to this function must be
   paired with a call to __resolv_context_put.  */
struct resolv_context *__resolv_context_get (void)
  __attribute__ ((warn_unused_result));
libc_hidden_proto (__resolv_context_get)

/* Deallocate the temporary resolver context.  Converse of
   __resolv_context_get.  Do nothing if CTX is NULL.  */
void __resolv_context_put (struct resolv_context *ctx);
libc_hidden_proto (__resolv_context_put)

/* Like __resolv_context_get, but the _res structure can be partially
   initialzed and those changes will not be overwritten.  */
struct resolv_context *__resolv_context_get_preinit (void)
  __attribute__ ((warn_unused_result));
libc_hidden_proto (__resolv_context_get_preinit)

/* Wrap a struct __res_state object in a struct resolv_context object.
   A call to this function must be paired with a call to
   __resolv_context_put.  */
struct resolv_context *__resolv_context_get_override (struct __res_state *)
  __attribute__ ((nonnull (1), warn_unused_result));
libc_hidden_proto (__resolv_context_get_override)

/* Return the search path entry at INDEX, or NULL if there are fewer
   than INDEX entries.  */
static __attribute__ ((nonnull (1), unused)) const char *
__resolv_context_search_list (const struct resolv_context *ctx, size_t index)
{
  if (ctx->conf != NULL)
    {
      if (index < ctx->conf->search_list_size)
        return ctx->conf->search_list[index];
      else
        return NULL;
    }
  /* Fallback.  ctx->resp->dnsrch is a NULL-terminated array.  */
  for (size_t i = 0; ctx->resp->dnsrch[i] != NULL && i < MAXDNSRCH; ++i)
    if (i == index)
      return ctx->resp->dnsrch[i];
  return NULL;
}

/* Return the number of name servers.  */
static __attribute__ ((nonnull (1), unused)) size_t
__resolv_context_nameserver_count (const struct resolv_context *ctx)
{
  if (ctx->conf != NULL)
    return ctx->conf->nameserver_list_size;
  else
    return ctx->resp->nscount;
}

/* Return a pointer to the socket address of the name server INDEX, or
   NULL if the index is out of bounds.  */
static __attribute__ ((nonnull (1), unused)) const struct sockaddr *
__resolv_context_nameserver (const struct resolv_context *ctx, size_t index)
{
  if (ctx->conf != NULL)
    {
      if (index < ctx->conf->nameserver_list_size)
        return ctx->conf->nameserver_list[index];
    }
  else
    if (index < ctx->resp->nscount)
      {
        if (ctx->resp->nsaddr_list[index].sin_family != 0)
          return (const struct sockaddr *) &ctx->resp->nsaddr_list[index];
        else
          return (const struct sockaddr *) &ctx->resp->_u._ext.nsaddrs[index];
      }
  return NULL;
}

/* Return the number of sort list entries.  */
static __attribute__ ((nonnull (1), unused)) size_t
__resolv_context_sort_count (const struct resolv_context *ctx)
{
  if (ctx->conf != NULL)
    return ctx->conf->sort_list_size;
  else
    return ctx->resp->nsort;
}

/* Return the sort list entry at INDEX.  */
static __attribute__ ((nonnull (1), unused)) struct resolv_sortlist_entry
__resolv_context_sort_entry (const struct resolv_context *ctx, size_t index)
{
  if (ctx->conf != NULL)
    {
      if (index < ctx->conf->sort_list_size)
        return ctx->conf->sort_list[index];
      /* Fall through.  */
    }
  else if (index < ctx->resp->nsort)
    return (struct resolv_sortlist_entry)
      {
        .addr = ctx->resp->sort_list[index].addr,
        .mask = ctx->resp->sort_list[index].mask,
      };

  return (struct resolv_sortlist_entry) { .mask = 0, };
}

/* Called during thread shutdown to free the associated resolver
   context (mostly in response to cancellation, otherwise the
   __resolv_context_get/__resolv_context_put pairing will already have
   deallocated the context object).  */
void __resolv_context_freeres (void) attribute_hidden;

#endif /* _RESOLV_CONTEXT_H */
