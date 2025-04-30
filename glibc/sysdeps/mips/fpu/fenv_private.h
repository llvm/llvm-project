/* Internal math stuff.  MIPS version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef MIPS_FENV_PRIVATE_H
#define MIPS_FENV_PRIVATE_H 1

/* Inline functions to speed up the math library implementation.  The
   default versions of these routines are in generic/fenv_private.h
   and call fesetround, feholdexcept, etc.  These routines use inlined
   code instead.  */

#include <fenv.h>
#include <fenv_libc.h>
#include <fpu_control.h>

#define _FPU_MASK_ALL (_FPU_MASK_V | _FPU_MASK_Z | _FPU_MASK_O \
		       |_FPU_MASK_U | _FPU_MASK_I | FE_ALL_EXCEPT)

static __always_inline void
libc_feholdexcept_mips (fenv_t *envp)
{
  fpu_control_t cw;

  /* Save the current state.  */
  _FPU_GETCW (cw);
  envp->__fp_control_register = cw;

  /* Clear all exception enable bits and flags.  */
  cw &= ~(_FPU_MASK_ALL);
  _FPU_SETCW (cw);
}
#define libc_feholdexcept libc_feholdexcept_mips
#define libc_feholdexceptf libc_feholdexcept_mips
#define libc_feholdexceptl libc_feholdexcept_mips

static __always_inline void
libc_fesetround_mips (int round)
{
  fpu_control_t cw;

  /* Get current state.  */
  _FPU_GETCW (cw);

  /* Set rounding bits.  */
  cw &= ~_FPU_RC_MASK;
  cw |= round;

  /* Set new state.  */
  _FPU_SETCW (cw);
}
#define libc_fesetround libc_fesetround_mips
#define libc_fesetroundf libc_fesetround_mips
#define libc_fesetroundl libc_fesetround_mips

static __always_inline void
libc_feholdexcept_setround_mips (fenv_t *envp, int round)
{
  fpu_control_t cw;

  /* Save the current state.  */
  _FPU_GETCW (cw);
  envp->__fp_control_register = cw;

  /* Clear all exception enable bits and flags.  */
  cw &= ~(_FPU_MASK_ALL);

  /* Set rounding bits.  */
  cw &= ~_FPU_RC_MASK;
  cw |= round;

  /* Set new state.  */
  _FPU_SETCW (cw);
}
#define libc_feholdexcept_setround libc_feholdexcept_setround_mips
#define libc_feholdexcept_setroundf libc_feholdexcept_setround_mips
#define libc_feholdexcept_setroundl libc_feholdexcept_setround_mips

#define libc_feholdsetround libc_feholdexcept_setround_mips
#define libc_feholdsetroundf libc_feholdexcept_setround_mips
#define libc_feholdsetroundl libc_feholdexcept_setround_mips

static __always_inline void
libc_fesetenv_mips (fenv_t *envp)
{
  fpu_control_t cw __attribute__ ((unused));

  /* Read current state to flush fpu pipeline.  */
  _FPU_GETCW (cw);

  _FPU_SETCW (envp->__fp_control_register);
}
#define libc_fesetenv libc_fesetenv_mips
#define libc_fesetenvf libc_fesetenv_mips
#define libc_fesetenvl libc_fesetenv_mips

static __always_inline int
libc_feupdateenv_test_mips (fenv_t *envp, int excepts)
{
  /* int ret = fetestexcept (excepts); feupdateenv (envp); return ret; */
  int cw, temp;

  /* Get current control word.  */
  _FPU_GETCW (cw);

  /* Set flag bits (which are accumulative), and *also* set the
     cause bits.  The setting of the cause bits is what actually causes
     the hardware to generate the exception, if the corresponding enable
     bit is set as well.  */
  temp = cw & FE_ALL_EXCEPT;
  temp |= envp->__fp_control_register | (temp << CAUSE_SHIFT);

  /* Set new state.  */
  _FPU_SETCW (temp);

  return cw & excepts & FE_ALL_EXCEPT;
}
#define libc_feupdateenv_test libc_feupdateenv_test_mips
#define libc_feupdateenv_testf libc_feupdateenv_test_mips
#define libc_feupdateenv_testl libc_feupdateenv_test_mips

static __always_inline void
libc_feupdateenv_mips (fenv_t *envp)
{
  libc_feupdateenv_test_mips (envp, 0);
}
#define libc_feupdateenv libc_feupdateenv_mips
#define libc_feupdateenvf libc_feupdateenv_mips
#define libc_feupdateenvl libc_feupdateenv_mips

#define libc_feresetround libc_feupdateenv_mips
#define libc_feresetroundf libc_feupdateenv_mips
#define libc_feresetroundl libc_feupdateenv_mips

static __always_inline int
libc_fetestexcept_mips (int excepts)
{
  int cw;

  /* Get current control word.  */
  _FPU_GETCW (cw);

  return cw & excepts & FE_ALL_EXCEPT;
}
#define libc_fetestexcept libc_fetestexcept_mips
#define libc_fetestexceptf libc_fetestexcept_mips
#define libc_fetestexceptl libc_fetestexcept_mips

/*  Enable support for rounding mode context.  */
#define HAVE_RM_CTX 1

static __always_inline void
libc_feholdexcept_setround_mips_ctx (struct rm_ctx *ctx, int round)
{
  fpu_control_t old, new;

  /* Save the current state.  */
  _FPU_GETCW (old);
  ctx->env.__fp_control_register = old;

  /* Clear all exception enable bits and flags.  */
  new = old & ~(_FPU_MASK_ALL);

  /* Set rounding bits.  */
  new = (new & ~_FPU_RC_MASK) | round;

  if (__glibc_unlikely (new != old))
    {
      _FPU_SETCW (new);
      ctx->updated_status = true;
    }
  else
    ctx->updated_status = false;
}
#define libc_feholdexcept_setround_ctx   libc_feholdexcept_setround_mips_ctx
#define libc_feholdexcept_setroundf_ctx  libc_feholdexcept_setround_mips_ctx
#define libc_feholdexcept_setroundl_ctx  libc_feholdexcept_setround_mips_ctx

static __always_inline void
libc_fesetenv_mips_ctx (struct rm_ctx *ctx)
{
  libc_fesetenv_mips (&ctx->env);
}
#define libc_fesetenv_ctx                libc_fesetenv_mips_ctx
#define libc_fesetenvf_ctx               libc_fesetenv_mips_ctx
#define libc_fesetenvl_ctx               libc_fesetenv_mips_ctx

static __always_inline void
libc_feupdateenv_mips_ctx (struct rm_ctx *ctx)
{
  if (__glibc_unlikely (ctx->updated_status))
    libc_feupdateenv_test_mips (&ctx->env, 0);
}
#define libc_feupdateenv_ctx             libc_feupdateenv_mips_ctx
#define libc_feupdateenvf_ctx            libc_feupdateenv_mips_ctx
#define libc_feupdateenvl_ctx            libc_feupdateenv_mips_ctx
#define libc_feresetround_ctx            libc_feupdateenv_mips_ctx
#define libc_feresetroundf_ctx           libc_feupdateenv_mips_ctx
#define libc_feresetroundl_ctx           libc_feupdateenv_mips_ctx

static __always_inline void
libc_feholdsetround_mips_ctx (struct rm_ctx *ctx, int round)
{
  fpu_control_t old, new;

  /* Save the current state.  */
  _FPU_GETCW (old);
  ctx->env.__fp_control_register = old;

  /* Set rounding bits.  */
  new = (old & ~_FPU_RC_MASK) | round;

  if (__glibc_unlikely (new != old))
    {
      _FPU_SETCW (new);
      ctx->updated_status = true;
    }
  else
    ctx->updated_status = false;
}
#define libc_feholdsetround_ctx          libc_feholdsetround_mips_ctx
#define libc_feholdsetroundf_ctx         libc_feholdsetround_mips_ctx
#define libc_feholdsetroundl_ctx         libc_feholdsetround_mips_ctx

#include_next <fenv_private.h>

#endif
