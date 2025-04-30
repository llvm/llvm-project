/* Private floating point rounding and exceptions handling.  AArch64 version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef AARCH64_FENV_PRIVATE_H
#define AARCH64_FENV_PRIVATE_H 1

#include <fenv.h>
#include <fpu_control.h>

static __always_inline void
libc_feholdexcept_aarch64 (fenv_t *envp)
{
  fpu_control_t fpcr;
  fpu_control_t new_fpcr;
  fpu_fpsr_t fpsr;
  fpu_fpsr_t new_fpsr;

  _FPU_GETCW (fpcr);
  _FPU_GETFPSR (fpsr);
  envp->__fpcr = fpcr;
  envp->__fpsr = fpsr;

  /* Clear exception flags and set all exceptions to non-stop.  */
  new_fpcr = fpcr & ~(FE_ALL_EXCEPT << FE_EXCEPT_SHIFT);
  new_fpsr = fpsr & ~FE_ALL_EXCEPT;

  if (__glibc_unlikely (new_fpcr != fpcr))
    _FPU_SETCW (new_fpcr);

  if (new_fpsr != fpsr)
    _FPU_SETFPSR (new_fpsr);
}

#define libc_feholdexcept  libc_feholdexcept_aarch64
#define libc_feholdexceptf libc_feholdexcept_aarch64
#define libc_feholdexceptl libc_feholdexcept_aarch64

static __always_inline void
libc_fesetround_aarch64 (int round)
{
  fpu_control_t fpcr;

  _FPU_GETCW (fpcr);

  /* Check whether rounding modes are different.  */
  round = (fpcr ^ round) & _FPU_FPCR_RM_MASK;

  /* Set new rounding mode if different.  */
  if (__glibc_unlikely (round != 0))
    _FPU_SETCW (fpcr ^ round);
}

#define libc_fesetround  libc_fesetround_aarch64
#define libc_fesetroundf libc_fesetround_aarch64
#define libc_fesetroundl libc_fesetround_aarch64

static __always_inline void
libc_feholdexcept_setround_aarch64 (fenv_t *envp, int round)
{
  fpu_control_t fpcr;
  fpu_control_t new_fpcr;
  fpu_fpsr_t fpsr;
  fpu_fpsr_t new_fpsr;

  _FPU_GETCW (fpcr);
  _FPU_GETFPSR (fpsr);
  envp->__fpcr = fpcr;
  envp->__fpsr = fpsr;

  /* Clear exception flags, set all exceptions to non-stop,
     and set new rounding mode.  */
  new_fpcr = fpcr & ~((FE_ALL_EXCEPT << FE_EXCEPT_SHIFT) | _FPU_FPCR_RM_MASK);
  new_fpcr |= round;
  new_fpsr = fpsr & ~FE_ALL_EXCEPT;

  if (__glibc_unlikely (new_fpcr != fpcr))
    _FPU_SETCW (new_fpcr);

  if (new_fpsr != fpsr)
    _FPU_SETFPSR (new_fpsr);
}

#define libc_feholdexcept_setround  libc_feholdexcept_setround_aarch64
#define libc_feholdexcept_setroundf libc_feholdexcept_setround_aarch64
#define libc_feholdexcept_setroundl libc_feholdexcept_setround_aarch64

static __always_inline int
libc_fetestexcept_aarch64 (int ex)
{
  fpu_fpsr_t fpsr;

  _FPU_GETFPSR (fpsr);
  return fpsr & ex & FE_ALL_EXCEPT;
}

#define libc_fetestexcept  libc_fetestexcept_aarch64
#define libc_fetestexceptf libc_fetestexcept_aarch64
#define libc_fetestexceptl libc_fetestexcept_aarch64

static __always_inline void
libc_fesetenv_aarch64 (const fenv_t *envp)
{
  fpu_control_t fpcr;
  fpu_control_t new_fpcr;

  _FPU_GETCW (fpcr);
  new_fpcr = envp->__fpcr;

  if (__glibc_unlikely (fpcr != new_fpcr))
    _FPU_SETCW (new_fpcr);

  _FPU_SETFPSR (envp->__fpsr);
}

#define libc_fesetenv  libc_fesetenv_aarch64
#define libc_fesetenvf libc_fesetenv_aarch64
#define libc_fesetenvl libc_fesetenv_aarch64
#define libc_feresetround_noex  libc_fesetenv_aarch64
#define libc_feresetround_noexf libc_fesetenv_aarch64
#define libc_feresetround_noexl libc_fesetenv_aarch64

static __always_inline int
libc_feupdateenv_test_aarch64 (const fenv_t *envp, int ex)
{
  fpu_control_t fpcr;
  fpu_control_t new_fpcr;
  fpu_fpsr_t fpsr;
  fpu_fpsr_t new_fpsr;
  int excepts;

  _FPU_GETCW (fpcr);
  _FPU_GETFPSR (fpsr);

  /* Merge current exception flags with the saved fenv.  */
  excepts = fpsr & FE_ALL_EXCEPT;
  new_fpcr = envp->__fpcr;
  new_fpsr = envp->__fpsr | excepts;

  if (__glibc_unlikely (fpcr != new_fpcr))
    _FPU_SETCW (new_fpcr);

  if (fpsr != new_fpsr)
    _FPU_SETFPSR (new_fpsr);

  /* Raise the exceptions if enabled in the new FP state.  */
  if (__glibc_unlikely (excepts & (new_fpcr >> FE_EXCEPT_SHIFT)))
    __feraiseexcept (excepts);

  return excepts & ex;
}

#define libc_feupdateenv_test  libc_feupdateenv_test_aarch64
#define libc_feupdateenv_testf libc_feupdateenv_test_aarch64
#define libc_feupdateenv_testl libc_feupdateenv_test_aarch64

static __always_inline void
libc_feupdateenv_aarch64 (const fenv_t *envp)
{
  libc_feupdateenv_test_aarch64 (envp, 0);
}

#define libc_feupdateenv  libc_feupdateenv_aarch64
#define libc_feupdateenvf libc_feupdateenv_aarch64
#define libc_feupdateenvl libc_feupdateenv_aarch64

static __always_inline void
libc_feholdsetround_aarch64 (fenv_t *envp, int round)
{
  fpu_control_t fpcr;
  fpu_fpsr_t fpsr;

  _FPU_GETCW (fpcr);
  _FPU_GETFPSR (fpsr);
  envp->__fpcr = fpcr;
  envp->__fpsr = fpsr;

  /* Check whether rounding modes are different.  */
  round = (fpcr ^ round) & _FPU_FPCR_RM_MASK;

  /* Set new rounding mode if different.  */
  if (__glibc_unlikely (round != 0))
    _FPU_SETCW (fpcr ^ round);
}

#define libc_feholdsetround  libc_feholdsetround_aarch64
#define libc_feholdsetroundf libc_feholdsetround_aarch64
#define libc_feholdsetroundl libc_feholdsetround_aarch64

static __always_inline void
libc_feresetround_aarch64 (fenv_t *envp)
{
  fpu_control_t fpcr;
  int round;

  _FPU_GETCW (fpcr);

  /* Check whether rounding modes are different.  */
  round = (envp->__fpcr ^ fpcr) & _FPU_FPCR_RM_MASK;

  /* Restore the rounding mode if it was changed.  */
  if (__glibc_unlikely (round != 0))
    _FPU_SETCW (fpcr ^ round);
}

#define libc_feresetround  libc_feresetround_aarch64
#define libc_feresetroundf libc_feresetround_aarch64
#define libc_feresetroundl libc_feresetround_aarch64

/* We have support for rounding mode context.  */
#define HAVE_RM_CTX 1

static __always_inline void
libc_feholdsetround_aarch64_ctx (struct rm_ctx *ctx, int r)
{
  fpu_control_t fpcr;
  int round;

  _FPU_GETCW (fpcr);
  ctx->env.__fpcr = fpcr;

  /* Check whether rounding modes are different.  */
  round = (fpcr ^ r) & _FPU_FPCR_RM_MASK;
  ctx->updated_status = round != 0;

  /* Set the rounding mode if changed.  */
  if (__glibc_unlikely (round != 0))
    _FPU_SETCW (fpcr ^ round);
}

#define libc_feholdsetround_ctx		libc_feholdsetround_aarch64_ctx
#define libc_feholdsetroundf_ctx	libc_feholdsetround_aarch64_ctx
#define libc_feholdsetroundl_ctx	libc_feholdsetround_aarch64_ctx

static __always_inline void
libc_feresetround_aarch64_ctx (struct rm_ctx *ctx)
{
  /* Restore the rounding mode if updated.  */
  if (__glibc_unlikely (ctx->updated_status))
    _FPU_SETCW (ctx->env.__fpcr);
}

#define libc_feresetround_ctx		libc_feresetround_aarch64_ctx
#define libc_feresetroundf_ctx		libc_feresetround_aarch64_ctx
#define libc_feresetroundl_ctx		libc_feresetround_aarch64_ctx

static __always_inline void
libc_feholdsetround_noex_aarch64_ctx (struct rm_ctx *ctx, int r)
{
  fpu_control_t fpcr;
  fpu_fpsr_t fpsr;
  int round;

  _FPU_GETCW (fpcr);
  _FPU_GETFPSR (fpsr);
  ctx->env.__fpcr = fpcr;
  ctx->env.__fpsr = fpsr;

  /* Check whether rounding modes are different.  */
  round = (fpcr ^ r) & _FPU_FPCR_RM_MASK;
  ctx->updated_status = round != 0;

  /* Set the rounding mode if changed.  */
  if (__glibc_unlikely (round != 0))
    _FPU_SETCW (fpcr ^ round);
}

#define libc_feholdsetround_noex_ctx	libc_feholdsetround_noex_aarch64_ctx
#define libc_feholdsetround_noexf_ctx	libc_feholdsetround_noex_aarch64_ctx
#define libc_feholdsetround_noexl_ctx	libc_feholdsetround_noex_aarch64_ctx

static __always_inline void
libc_feresetround_noex_aarch64_ctx (struct rm_ctx *ctx)
{
  /* Restore the rounding mode if updated.  */
  if (__glibc_unlikely (ctx->updated_status))
    _FPU_SETCW (ctx->env.__fpcr);

  /* Write new FPSR to restore exception flags.  */
  _FPU_SETFPSR (ctx->env.__fpsr);
}

#define libc_feresetround_noex_ctx	libc_feresetround_noex_aarch64_ctx
#define libc_feresetround_noexf_ctx	libc_feresetround_noex_aarch64_ctx
#define libc_feresetround_noexl_ctx	libc_feresetround_noex_aarch64_ctx

#include_next <fenv_private.h>

#endif
