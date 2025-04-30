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

#include <resolv_context.h>
#include <resolv_conf.h>
#include <resolv-internal.h>

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

/* Currently active struct resolv_context object.  This pointer forms
   the start of a single-linked list, using the __next member of
   struct resolv_context.  This list serves two purposes:

   (a) A subsequent call to __resolv_context_get will only increment
       the reference counter and will not allocate a new object.  The
       _res state freshness check is skipped in this case, too.

   (b) The per-thread cleanup function defined by the resolver calls
       __resolv_context_freeres, which will deallocate all the context
       objects.  This avoids the need for cancellation handlers and
       the complexity they bring, but it requires heap allocation of
       the context object because the per-thread cleanup functions run
       only after the stack has been fully unwound (and all on-stack
       objects have been deallocated at this point).

   The TLS variable current is updated even in
   __resolv_context_get_override, to support case (b) above.  This does
   not override the per-thread resolver state (as obtained by the
   non-res_state function such as __resolv_context_get) in an
   observable way because the wrapped context is only used to
   implement the res_n* functions in the resolver, and those do not
   call back into user code which could indirectly use the per-thread
   resolver state.  */
static __thread struct resolv_context *current attribute_tls_model_ie;

/* The resolv_conf handling will gives us a ctx->conf pointer even if
   these fields do not match because a mis-match does not cause a loss
   of state (_res objects can store the full information).  This
   function checks to ensure that there is a full patch, to prevent
   overwriting a patched configuration.  */
static bool
replicated_configuration_matches (const struct resolv_context *ctx)
{
  return ctx->resp->options == ctx->conf->options
    && ctx->resp->retrans == ctx->conf->retrans
    && ctx->resp->retry == ctx->conf->retry
    && ctx->resp->ndots == ctx->conf->ndots;
}

/* Initialize *RESP if RES_INIT is not yet set in RESP->options, or if
   res_init in some other thread requested re-initializing.  */
static __attribute__ ((warn_unused_result)) bool
maybe_init (struct resolv_context *ctx, bool preinit)
{
  struct __res_state *resp = ctx->resp;
  if (resp->options & RES_INIT)
    {
      if (resp->options & RES_NORELOAD)
        /* Configuration reloading was explicitly disabled.  */
        return true;

      /* If there is no associated resolv_conf object despite the
         initialization, something modified *ctx->resp.  Do not
         override those changes.  */
      if (ctx->conf != NULL && replicated_configuration_matches (ctx))
        {
          struct resolv_conf *current = __resolv_conf_get_current ();
          if (current == NULL)
            return false;

          /* Check if the configuration changed.  */
          if (current != ctx->conf)
            {
              /* This call will detach the extended resolver state.  */
              if (resp->nscount > 0)
                __res_iclose (resp, true);
              /* Reattach the current configuration.  */
              if (__resolv_conf_attach (ctx->resp, current))
                {
                  __resolv_conf_put (ctx->conf);
                  /* ctx takes ownership, so we do not release current.  */
                  ctx->conf = current;
                }
            }
          else
            /* No change.  Drop the reference count for current.  */
            __resolv_conf_put (current);
        }
      return true;
    }

  assert (ctx->conf == NULL);
  if (preinit)
    {
      if (!resp->retrans)
        resp->retrans = RES_TIMEOUT;
      if (!resp->retry)
        resp->retry = RES_DFLRETRY;
      resp->options = RES_DEFAULT;
      if (!resp->id)
        resp->id = res_randomid ();
    }

  if (__res_vinit (resp, preinit) < 0)
    return false;
  ctx->conf = __resolv_conf_get (ctx->resp);
  return true;
}

/* Allocate a new context object and initialize it.  The object is put
   on the current list.  */
static struct resolv_context *
context_alloc (struct __res_state *resp)
{
  struct resolv_context *ctx = malloc (sizeof (*ctx));
  if (ctx == NULL)
    return NULL;
  ctx->resp = resp;
  ctx->conf = __resolv_conf_get (resp);
  ctx->__refcount = 1;
  ctx->__from_res = true;
  ctx->__next = current;
  current = ctx;
  return ctx;
}

/* Deallocate the context object and all the state within.   */
static void
context_free (struct resolv_context *ctx)
{
  int error_code = errno;
  current = ctx->__next;
  __resolv_conf_put (ctx->conf);
  free (ctx);
  __set_errno (error_code);
}

/* Reuse the current context object.  */
static struct resolv_context *
context_reuse (void)
{
  /* A context object created by __resolv_context_get_override cannot
     be reused.  */
  assert (current->__from_res);

  ++current->__refcount;

  /* Check for reference counter wraparound.  This can only happen if
     the get/put functions are not properly paired.  */
  assert (current->__refcount > 0);

  return current;
}

/* Backing function for the __resolv_context_get family of
   functions.  */
static struct resolv_context *
context_get (bool preinit)
{
  if (current != NULL)
    return context_reuse ();

  struct resolv_context *ctx = context_alloc (&_res);
  if (ctx == NULL)
    return NULL;
  if (!maybe_init (ctx, preinit))
    {
      context_free (ctx);
      return NULL;
    }
  return ctx;
}

struct resolv_context *
__resolv_context_get (void)
{
  return context_get (false);
}
libc_hidden_def (__resolv_context_get)

struct resolv_context *
__resolv_context_get_preinit (void)
{
  return context_get (true);
}
libc_hidden_def (__resolv_context_get_preinit)

struct resolv_context *
__resolv_context_get_override (struct __res_state *resp)
{
  /* NB: As explained asbove, context_alloc will put the context on
     the current list.  */
  struct resolv_context *ctx = context_alloc (resp);
  if (ctx == NULL)
    return NULL;

  ctx->__from_res = false;
  return ctx;
}
libc_hidden_def (__resolv_context_get_override)

void
__resolv_context_put (struct resolv_context *ctx)
{
  if (ctx == NULL)
    return;

  /* NB: Callers assume that this function preserves errno and
     h_errno.  */

  assert (current == ctx);
  assert (ctx->__refcount > 0);

  if (ctx->__from_res && --ctx->__refcount > 0)
    /* Do not pop this context yet.  */
    return;

  context_free (ctx);
}
libc_hidden_def (__resolv_context_put)

void
__resolv_context_freeres (void)
{
  /* Deallocate the entire chain of context objects.  */
  struct resolv_context *ctx = current;
  current = NULL;
  while (ctx != NULL)
    {
      struct resolv_context *next = ctx->__next;
      context_free (ctx);
      ctx = next;
    }
}
