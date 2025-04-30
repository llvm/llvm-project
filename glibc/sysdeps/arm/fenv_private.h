/* Private floating point rounding and exceptions handling.  ARM VFP version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef ARM_FENV_PRIVATE_H
#define ARM_FENV_PRIVATE_H 1

#include <fenv.h>
#include <fpu_control.h>

static __always_inline void
libc_feholdexcept_vfp (fenv_t *envp)
{
  fpu_control_t fpscr;

  _FPU_GETCW (fpscr);
  envp->__cw = fpscr;

  /* Clear exception flags and set all exceptions to non-stop.  */
  fpscr &= ~_FPU_MASK_EXCEPT;
  _FPU_SETCW (fpscr);
}

static __always_inline void
libc_fesetround_vfp (int round)
{
  fpu_control_t fpscr;

  _FPU_GETCW (fpscr);

  /* Set new rounding mode if different.  */
  if (__glibc_unlikely ((fpscr & _FPU_MASK_RM) != round))
    _FPU_SETCW ((fpscr & ~_FPU_MASK_RM) | round);
}

static __always_inline void
libc_feholdexcept_setround_vfp (fenv_t *envp, int round)
{
  fpu_control_t fpscr;

  _FPU_GETCW (fpscr);
  envp->__cw = fpscr;

  /* Clear exception flags, set all exceptions to non-stop,
     and set new rounding mode.  */
  fpscr &= ~(_FPU_MASK_EXCEPT | _FPU_MASK_RM);
  _FPU_SETCW (fpscr | round);
}

static __always_inline void
libc_feholdsetround_vfp (fenv_t *envp, int round)
{
  fpu_control_t fpscr;

  _FPU_GETCW (fpscr);
  envp->__cw = fpscr;

  /* Set new rounding mode if different.  */
  if (__glibc_unlikely ((fpscr & _FPU_MASK_RM) != round))
    _FPU_SETCW ((fpscr & ~_FPU_MASK_RM) | round);
}

static __always_inline void
libc_feresetround_vfp (fenv_t *envp)
{
  fpu_control_t fpscr, round;

  _FPU_GETCW (fpscr);

  /* Check whether rounding modes are different.  */
  round = (envp->__cw ^ fpscr) & _FPU_MASK_RM;

  /* Restore the rounding mode if it was changed.  */
  if (__glibc_unlikely (round != 0))
    _FPU_SETCW (fpscr ^ round);
}

static __always_inline int
libc_fetestexcept_vfp (int ex)
{
  fpu_control_t fpscr;

  _FPU_GETCW (fpscr);
  return fpscr & ex & FE_ALL_EXCEPT;
}

static __always_inline void
libc_fesetenv_vfp (const fenv_t *envp)
{
  fpu_control_t fpscr, new_fpscr;

  _FPU_GETCW (fpscr);
  new_fpscr = envp->__cw;

  /* Write new FPSCR if different (ignoring NZCV flags).  */
  if (__glibc_unlikely (((fpscr ^ new_fpscr) & ~_FPU_MASK_NZCV) != 0))
    _FPU_SETCW (new_fpscr);
}

static __always_inline int
libc_feupdateenv_test_vfp (const fenv_t *envp, int ex)
{
  fpu_control_t fpscr, new_fpscr;
  int excepts;

  _FPU_GETCW (fpscr);

  /* Merge current exception flags with the saved fenv.  */
  excepts = fpscr & FE_ALL_EXCEPT;
  new_fpscr = envp->__cw | excepts;

  /* Write new FPSCR if different (ignoring NZCV flags).  */
  if (__glibc_unlikely (((fpscr ^ new_fpscr) & ~_FPU_MASK_NZCV) != 0))
    _FPU_SETCW (new_fpscr);

  /* Raise the exceptions if enabled in the new FP state.  */
  if (__glibc_unlikely (excepts & (new_fpscr >> FE_EXCEPT_SHIFT)))
    __feraiseexcept (excepts);

  return excepts & ex;
}

static __always_inline void
libc_feupdateenv_vfp (const fenv_t *envp)
{
  libc_feupdateenv_test_vfp (envp, 0);
}

static __always_inline void
libc_feholdsetround_vfp_ctx (struct rm_ctx *ctx, int r)
{
  fpu_control_t fpscr, round;

  _FPU_GETCW (fpscr);
  ctx->updated_status = false;
  ctx->env.__cw = fpscr;

  /* Check whether rounding modes are different.  */
  round = (fpscr ^ r) & _FPU_MASK_RM;

  /* Set the rounding mode if changed.  */
  if (__glibc_unlikely (round != 0))
    {
      ctx->updated_status = true;
      _FPU_SETCW (fpscr ^ round);
    }
}

static __always_inline void
libc_feresetround_vfp_ctx (struct rm_ctx *ctx)
{
  /* Restore the rounding mode if updated.  */
  if (__glibc_unlikely (ctx->updated_status))
    {
      fpu_control_t fpscr;

      _FPU_GETCW (fpscr);
      fpscr = (fpscr & ~_FPU_MASK_RM) | (ctx->env.__cw & _FPU_MASK_RM);
      _FPU_SETCW (fpscr);
    }
}

static __always_inline void
libc_fesetenv_vfp_ctx (struct rm_ctx *ctx)
{
  fpu_control_t fpscr, new_fpscr;

  _FPU_GETCW (fpscr);
  new_fpscr = ctx->env.__cw;

  /* Write new FPSCR if different (ignoring NZCV flags).  */
  if (__glibc_unlikely (((fpscr ^ new_fpscr) & ~_FPU_MASK_NZCV) != 0))
    _FPU_SETCW (new_fpscr);
}

#ifndef __SOFTFP__

# define libc_feholdexcept  libc_feholdexcept_vfp
# define libc_feholdexceptf libc_feholdexcept_vfp
# define libc_feholdexceptl libc_feholdexcept_vfp

# define libc_fesetround  libc_fesetround_vfp
# define libc_fesetroundf libc_fesetround_vfp
# define libc_fesetroundl libc_fesetround_vfp

# define libc_feresetround  libc_feresetround_vfp
# define libc_feresetroundf libc_feresetround_vfp
# define libc_feresetroundl libc_feresetround_vfp

# define libc_feresetround_noex  libc_fesetenv_vfp
# define libc_feresetround_noexf libc_fesetenv_vfp
# define libc_feresetround_noexl libc_fesetenv_vfp

# define libc_feholdexcept_setround  libc_feholdexcept_setround_vfp
# define libc_feholdexcept_setroundf libc_feholdexcept_setround_vfp
# define libc_feholdexcept_setroundl libc_feholdexcept_setround_vfp

# define libc_feholdsetround  libc_feholdsetround_vfp
# define libc_feholdsetroundf libc_feholdsetround_vfp
# define libc_feholdsetroundl libc_feholdsetround_vfp

# define libc_fetestexcept  libc_fetestexcept_vfp
# define libc_fetestexceptf libc_fetestexcept_vfp
# define libc_fetestexceptl libc_fetestexcept_vfp

# define libc_fesetenv  libc_fesetenv_vfp
# define libc_fesetenvf libc_fesetenv_vfp
# define libc_fesetenvl libc_fesetenv_vfp

# define libc_feupdateenv  libc_feupdateenv_vfp
# define libc_feupdateenvf libc_feupdateenv_vfp
# define libc_feupdateenvl libc_feupdateenv_vfp

# define libc_feupdateenv_test  libc_feupdateenv_test_vfp
# define libc_feupdateenv_testf libc_feupdateenv_test_vfp
# define libc_feupdateenv_testl libc_feupdateenv_test_vfp

/* We have support for rounding mode context.  */
#define HAVE_RM_CTX 1

# define libc_feholdsetround_ctx	libc_feholdsetround_vfp_ctx
# define libc_feresetround_ctx		libc_feresetround_vfp_ctx
# define libc_feresetround_noex_ctx	libc_fesetenv_vfp_ctx

# define libc_feholdsetroundf_ctx	libc_feholdsetround_vfp_ctx
# define libc_feresetroundf_ctx		libc_feresetround_vfp_ctx
# define libc_feresetround_noexf_ctx	libc_fesetenv_vfp_ctx

# define libc_feholdsetroundl_ctx	libc_feholdsetround_vfp_ctx
# define libc_feresetroundl_ctx		libc_feresetround_vfp_ctx
# define libc_feresetround_noexl_ctx	libc_fesetenv_vfp_ctx

#endif

#include_next <fenv_private.h>

#endif /* ARM_FENV_PRIVATE_H */
