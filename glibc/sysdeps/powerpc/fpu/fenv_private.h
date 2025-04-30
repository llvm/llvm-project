/* Private floating point rounding and exceptions handling. PowerPC version.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef POWERPC_FENV_PRIVATE_H
#define POWERPC_FENV_PRIVATE_H 1

#include <fenv.h>
#include <fenv_libc.h>
#include <fpu_control.h>

#ifdef _ARCH_PWR8
/* There is no performance advantage to non-stop mode.  */
/* The odd syntax here is to innocuously reference the given variables
   to prevent warnings about unused variables.  */
#define __TEST_AND_BEGIN_NON_STOP(old, new) do {} while ((old) * (new) * 0 != 0)
#define __TEST_AND_END_NON_STOP(old, new) do {} while ((old) * (new) * 0 != 0)
#else
#define __TEST_AND_BEGIN_NON_STOP __TEST_AND_ENTER_NON_STOP
#define __TEST_AND_END_NON_STOP __TEST_AND_EXIT_NON_STOP
#endif

static __always_inline void
libc_feholdexcept_setround_ppc (fenv_t *envp, int r)
{
  fenv_union_t old, new;

  old.fenv = *envp = fegetenv_register ();

  __TEST_AND_BEGIN_NON_STOP (old.l, 0ULL);

  /* Clear everything and set the rounding mode.  */
  new.l = r;
  fesetenv_register (new.fenv);
}

static __always_inline unsigned long long
__libc_femergeenv_ppc (const fenv_t *envp, unsigned long long old_mask,
	unsigned long long new_mask)
{
  fenv_union_t old, new;

  new.fenv = *envp;
  old.fenv = fegetenv_register ();

  /* Merge bits while masking unwanted bits from new and old env.  */
  new.l = (old.l & old_mask) | (new.l & new_mask);

  __TEST_AND_END_NON_STOP (old.l, new.l);
  __TEST_AND_BEGIN_NON_STOP (old.l, new.l);

  /* If requesting to keep status, replace control, and merge exceptions,
     and exceptions haven't changed, we can just set new control instead
     of the whole FPSCR.  */
  if ((old_mask & (FPSCR_CONTROL_MASK|FPSCR_STATUS_MASK|FPSCR_EXCEPTIONS_MASK))
      == (FPSCR_STATUS_MASK|FPSCR_EXCEPTIONS_MASK) &&
      (new_mask & (FPSCR_CONTROL_MASK|FPSCR_STATUS_MASK|FPSCR_EXCEPTIONS_MASK))
      == (FPSCR_CONTROL_MASK|FPSCR_EXCEPTIONS_MASK) &&
      (old.l & FPSCR_EXCEPTIONS_MASK) == (new.l & FPSCR_EXCEPTIONS_MASK))
  {
    fesetenv_control (new.fenv);
  }
  else
    /* Atomically enable and raise (if appropriate) exceptions set in `new'.  */
    fesetenv_register (new.fenv);

  return old.l;
}

static __always_inline void
libc_fesetenv_ppc (const fenv_t *envp)
{
  /* Replace the entire environment.  */
  __libc_femergeenv_ppc (envp, 0LL, -1LL);
}

static __always_inline void
libc_feresetround_ppc (fenv_t *envp)
{
  fenv_union_t new = { .fenv = *envp };
  fegetenv_and_set_rn (new.l & FPSCR_RN_MASK);
}

static __always_inline int
libc_feupdateenv_test_ppc (fenv_t *envp, int ex)
{
  return __libc_femergeenv_ppc (envp, ~FPSCR_CONTROL_MASK,
				~FPSCR_STATUS_MASK) & ex;
}

static __always_inline void
libc_feupdateenv_ppc (fenv_t *e)
{
  libc_feupdateenv_test_ppc (e, 0);
}

#define libc_feholdexceptf           libc_feholdexcept_ppc
#define libc_feholdexcept            libc_feholdexcept_ppc
#define libc_feholdexcept_setroundf  libc_feholdexcept_setround_ppc
#define libc_feholdexcept_setround   libc_feholdexcept_setround_ppc
#define libc_fetestexceptf           libc_fetestexcept_ppc
#define libc_fetestexcept            libc_fetestexcept_ppc
#define libc_fesetroundf             libc_fesetround_ppc
#define libc_fesetround              libc_fesetround_ppc
#define libc_fesetenvf               libc_fesetenv_ppc
#define libc_fesetenv                libc_fesetenv_ppc
#define libc_feupdateenv_testf       libc_feupdateenv_test_ppc
#define libc_feupdateenv_test        libc_feupdateenv_test_ppc
#define libc_feupdateenvf            libc_feupdateenv_ppc
#define libc_feupdateenv             libc_feupdateenv_ppc
#define libc_feholdsetroundf         libc_feholdsetround_ppc
#define libc_feholdsetround          libc_feholdsetround_ppc
#define libc_feresetroundf           libc_feresetround_ppc
#define libc_feresetround            libc_feresetround_ppc


/* We have support for rounding mode context.  */
#define HAVE_RM_CTX 1

static __always_inline void
libc_feholdsetround_ppc_ctx (struct rm_ctx *ctx, int r)
{
  fenv_union_t old;

  ctx->env = old.fenv = fegetenv_and_set_rn (r);
  ctx->updated_status = (r != (old.l & FPSCR_RN_MASK));
}

static __always_inline void
libc_feholdsetround_noex_ppc_ctx (struct rm_ctx *ctx, int r)
{
  fenv_union_t old, new;

  old.fenv = fegetenv_register ();

  new.l = (old.l & ~(FPSCR_ENABLES_MASK|FPSCR_RN_MASK)) | r;

  ctx->env = old.fenv;
  if (__glibc_unlikely (new.l != old.l))
    {
      __TEST_AND_BEGIN_NON_STOP (old.l, 0ULL);
      fesetenv_control (new.fenv);
      ctx->updated_status = true;
    }
  else
    ctx->updated_status = false;
}

static __always_inline void
libc_fesetenv_ppc_ctx (struct rm_ctx *ctx)
{
  libc_fesetenv_ppc (&ctx->env);
}

static __always_inline void
libc_feupdateenv_ppc_ctx (struct rm_ctx *ctx)
{
  if (__glibc_unlikely (ctx->updated_status))
    libc_feresetround_ppc (&ctx->env);
}

static __always_inline void
libc_feresetround_ppc_ctx (struct rm_ctx *ctx)
{
  if (__glibc_unlikely (ctx->updated_status))
    libc_feresetround_ppc (&ctx->env);
}

#define libc_fesetenv_ctx                libc_fesetenv_ppc_ctx
#define libc_fesetenvf_ctx               libc_fesetenv_ppc_ctx
#define libc_fesetenvl_ctx               libc_fesetenv_ppc_ctx
#define libc_feholdsetround_ctx          libc_feholdsetround_ppc_ctx
#define libc_feholdsetroundf_ctx         libc_feholdsetround_ppc_ctx
#define libc_feholdsetroundl_ctx         libc_feholdsetround_ppc_ctx
#define libc_feholdsetround_noex_ctx     libc_feholdsetround_noex_ppc_ctx
#define libc_feholdsetround_noexf_ctx    libc_feholdsetround_noex_ppc_ctx
#define libc_feholdsetround_noexl_ctx    libc_feholdsetround_noex_ppc_ctx
#define libc_feresetround_ctx            libc_feresetround_ppc_ctx
#define libc_feresetroundf_ctx           libc_feresetround_ppc_ctx
#define libc_feresetroundl_ctx           libc_feresetround_ppc_ctx
#define libc_feupdateenv_ctx             libc_feupdateenv_ppc_ctx
#define libc_feupdateenvf_ctx            libc_feupdateenv_ppc_ctx
#define libc_feupdateenvl_ctx            libc_feupdateenv_ppc_ctx

#include_next <fenv_private.h>

#endif
