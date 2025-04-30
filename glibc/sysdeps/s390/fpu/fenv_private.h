/* Private floating point rounding and exceptions handling.  390/s390x version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef S390_FENV_PRIVATE_H
#define S390_FENV_PRIVATE_H 1

#include <fenv.h>
#include <fenv_libc.h>
#include <fpu_control.h>

static __always_inline void
libc_feholdexcept_s390 (fenv_t *envp)
{
  fpu_control_t fpc, fpc_new;

  /* Store the environment.  */
  _FPU_GETCW (fpc);
  envp->__fpc = fpc;

  /* Clear the current exception flags and dxc field.
     Hold from generating fpu exceptions temporarily.  */
  fpc_new = fpc & ~(FPC_FLAGS_MASK | FPC_DXC_MASK | FPC_EXCEPTION_MASK);

  /* Only set new environment if it has changed.  */
  if (fpc_new != fpc)
    _FPU_SETCW (fpc_new);
}

#define libc_feholdexcept  libc_feholdexcept_s390
#define libc_feholdexceptf libc_feholdexcept_s390
#define libc_feholdexceptl libc_feholdexcept_s390

static __always_inline void
libc_fesetround_s390 (int round)
{
  __asm__ __volatile__ ("srnm 0(%0)" : : "a" (round));
}

#define libc_fesetround  libc_fesetround_s390
#define libc_fesetroundf libc_fesetround_s390
#define libc_fesetroundl libc_fesetround_s390

static __always_inline void
libc_feholdexcept_setround_s390 (fenv_t *envp, int r)
{
  fpu_control_t fpc, fpc_new;

  _FPU_GETCW (fpc);
  envp->__fpc = fpc;

  /* Clear the current exception flags and dxc field.
     Hold from generating fpu exceptions temporarily.
     Reset rounding mode bits.  */
  fpc_new = fpc & ~(FPC_FLAGS_MASK | FPC_DXC_MASK | FPC_EXCEPTION_MASK
		    | FPC_RM_MASK);

  /* Set new rounding mode.  */
  fpc_new |= (r & FPC_RM_MASK);

  /* Only set new environment if it has changed.  */
  if (fpc_new != fpc)
    _FPU_SETCW (fpc_new);
}

#define libc_feholdexcept_setround  libc_feholdexcept_setround_s390
#define libc_feholdexcept_setroundf libc_feholdexcept_setround_s390
#define libc_feholdexcept_setroundl libc_feholdexcept_setround_s390

static __always_inline int
libc_fetestexcept_s390 (int excepts)
{
  int res;
  fexcept_t fpc;

  _FPU_GETCW (fpc);

  /* Get current exceptions.  */
  res = (fpc >> FPC_FLAGS_SHIFT) & FE_ALL_EXCEPT;
  if ((fpc & FPC_NOT_FPU_EXCEPTION) == 0)
    /* Bits 6, 7 of dxc-byte are zero,
       thus bits 0-5 of dxc-byte correspond to the flag-bits.
       Evaluate flags and last dxc-exception-code.  */
    res |= (fpc >> FPC_DXC_SHIFT) & FE_ALL_EXCEPT;

  return res & excepts;
}

#define libc_fetestexcept  libc_fetestexcept_s390
#define libc_fetestexceptf libc_fetestexcept_s390
#define libc_fetestexceptl libc_fetestexcept_s390

static __always_inline void
libc_fesetenv_s390 (const fenv_t *envp)
{
  _FPU_SETCW (envp->__fpc);
}

#define libc_fesetenv  libc_fesetenv_s390
#define libc_fesetenvf libc_fesetenv_s390
#define libc_fesetenvl libc_fesetenv_s390

static __always_inline int
libc_feupdateenv_test_s390 (const fenv_t *envp, int ex)
{
  /* Get the currently raised exceptions.  */
  int excepts;
  fexcept_t fpc_old;

  _FPU_GETCW (fpc_old);

  /* Get current exceptions.  */
  excepts = (fpc_old >> FPC_FLAGS_SHIFT) & FE_ALL_EXCEPT;
  if ((fpc_old & FPC_NOT_FPU_EXCEPTION) == 0)
    /* Bits 6, 7 of dxc-byte are zero,
       thus bits 0-5 of dxc-byte correspond to the flag-bits.
       Evaluate flags and last dxc-exception-code.  */
    excepts |= (fpc_old >> FPC_DXC_SHIFT) & FE_ALL_EXCEPT;

  /* Merge the currently raised exceptions with those in envp.  */
  fpu_control_t fpc_new = envp->__fpc;
  fpc_new |= excepts << FPC_FLAGS_SHIFT;

  /* Install the new fpc from envp.  */
  if (fpc_new != fpc_old)
    _FPU_SETCW (fpc_new);

  /* Raise the exceptions if enabled in new fpc.  */
  if (__glibc_unlikely ((fpc_new >> FPC_EXCEPTION_MASK_SHIFT) & excepts))
    __feraiseexcept (excepts);

  return excepts & ex;
}

#define libc_feupdateenv_test  libc_feupdateenv_test_s390
#define libc_feupdateenv_testf libc_feupdateenv_test_s390
#define libc_feupdateenv_testl libc_feupdateenv_test_s390

static __always_inline void
libc_feupdateenv_s390 (const fenv_t *envp)
{
  libc_feupdateenv_test_s390 (envp, 0);
}

#define libc_feupdateenv  libc_feupdateenv_s390
#define libc_feupdateenvf libc_feupdateenv_s390
#define libc_feupdateenvl libc_feupdateenv_s390

static __always_inline fenv_t
libc_handle_user_fenv_s390 (const fenv_t *envp)
{
  fenv_t env;
  if (envp == FE_DFL_ENV)
    {
      env.__fpc = _FPU_DEFAULT;
    }
  else if (envp == FE_NOMASK_ENV)
    {
      env.__fpc = FPC_EXCEPTION_MASK;
    }
  else
    env = (*envp);

  return env;
}

/* We have support for rounding mode context.  */
#define HAVE_RM_CTX 1

static __always_inline void
libc_feholdsetround_s390_ctx (struct rm_ctx *ctx, int r)
{
  fpu_control_t fpc;
  int round;

  _FPU_GETCW (fpc);
  ctx->env.__fpc = fpc;

  /* Check whether rounding modes are different.  */
  round = fpc & FPC_RM_MASK;

  /* Set the rounding mode if changed.  */
  if (__glibc_unlikely (round != r))
    {
      ctx->updated_status = true;
      libc_fesetround_s390 (r);
    }
  else
    ctx->updated_status = false;
}

#define libc_feholdsetround_ctx		libc_feholdsetround_s390_ctx
#define libc_feholdsetroundf_ctx	libc_feholdsetround_s390_ctx
#define libc_feholdsetroundl_ctx	libc_feholdsetround_s390_ctx

static __always_inline void
libc_feresetround_s390_ctx (struct rm_ctx *ctx)
{
  /* Restore the rounding mode if updated.  */
  if (__glibc_unlikely (ctx->updated_status))
    {
      fpu_control_t fpc;
      _FPU_GETCW (fpc);
      fpc = ctx->env.__fpc | (fpc & FPC_FLAGS_MASK);
      _FPU_SETCW (fpc);
    }
}

#define libc_feresetround_ctx		libc_feresetround_s390_ctx
#define libc_feresetroundf_ctx		libc_feresetround_s390_ctx
#define libc_feresetroundl_ctx		libc_feresetround_s390_ctx

static __always_inline void
libc_feholdsetround_noex_s390_ctx (struct rm_ctx *ctx, int r)
{
  libc_feholdexcept_setround_s390 (&ctx->env, r);
}

#define libc_feholdsetround_noex_ctx	libc_feholdsetround_noex_s390_ctx
#define libc_feholdsetround_noexf_ctx	libc_feholdsetround_noex_s390_ctx
#define libc_feholdsetround_noexl_ctx	libc_feholdsetround_noex_s390_ctx

static __always_inline void
libc_feresetround_noex_s390_ctx (struct rm_ctx *ctx)
{
  /* Restore exception flags and rounding mode.  */
  libc_fesetenv_s390 (&ctx->env);
}

#define libc_feresetround_noex_ctx	libc_feresetround_noex_s390_ctx
#define libc_feresetround_noexf_ctx	libc_feresetround_noex_s390_ctx
#define libc_feresetround_noexl_ctx	libc_feresetround_noex_s390_ctx

#include_next <fenv_private.h>

#endif
