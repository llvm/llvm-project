/* Private floating point rounding and exceptions handling.  RISC-V version.
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

#ifndef RISCV_FENV_PRIVATE_H
#define RISCV_FENV_PRIVATE_H 1

#include <fenv.h>
#include <fpu_control.h>
#include <get-rounding-mode.h>

static __always_inline int
riscv_getround (void)
{
  return get_rounding_mode ();
}

static __always_inline void
riscv_setround (int rm)
{
  asm volatile ("fsrm %z0" : : "rJ" (rm));
}

static __always_inline int
riscv_getflags (void)
{
  int flags;
  asm volatile ("frflags %0" : "=r" (flags));
  return flags;
}

static __always_inline void
riscv_setflags (int flags)
{
  asm volatile ("fsflags %z0" : : "rJ" (flags));
}

static __always_inline void
libc_feholdexcept_riscv (fenv_t *envp)
{
  asm volatile ("csrrc %0, fcsr, %1" : "=r" (*envp) : "i" (FE_ALL_EXCEPT));
}

#define libc_feholdexcept  libc_feholdexcept_riscv
#define libc_feholdexceptf libc_feholdexcept_riscv
#define libc_feholdexceptl libc_feholdexcept_riscv

static __always_inline void
libc_fesetround_riscv (int round)
{
  riscv_setround (round);
}

#define libc_fesetround  libc_fesetround_riscv
#define libc_fesetroundf libc_fesetround_riscv
#define libc_fesetroundl libc_fesetround_riscv

static __always_inline void
libc_feholdexcept_setround_riscv (fenv_t *envp, int round)
{
  libc_feholdexcept_riscv (envp);
  libc_fesetround_riscv (round);
}

#define libc_feholdexcept_setround  libc_feholdexcept_setround_riscv
#define libc_feholdexcept_setroundf libc_feholdexcept_setround_riscv
#define libc_feholdexcept_setroundl libc_feholdexcept_setround_riscv

static __always_inline int
libc_fetestexcept_riscv (int ex)
{
  return riscv_getflags () & ex;
}

#define libc_fetestexcept  libc_fetestexcept_riscv
#define libc_fetestexceptf libc_fetestexcept_riscv
#define libc_fetestexceptl libc_fetestexcept_riscv

static __always_inline void
libc_fesetenv_riscv (const fenv_t *envp)
{
  long int env = (long int) envp - (long int) FE_DFL_ENV;
  if (env != 0)
    env = *envp;

  _FPU_SETCW (env);
}

#define libc_fesetenv  libc_fesetenv_riscv
#define libc_fesetenvf libc_fesetenv_riscv
#define libc_fesetenvl libc_fesetenv_riscv
#define libc_feresetround_noex  libc_fesetenv_riscv
#define libc_feresetround_noexf libc_fesetenv_riscv
#define libc_feresetround_noexl libc_fesetenv_riscv

static __always_inline int
libc_feupdateenv_test_riscv (const fenv_t *envp, int ex)
{
  fenv_t env = *envp;
  int flags = riscv_getflags ();
  asm volatile ("csrw fcsr, %z0" : : "rJ" (env | flags));
  return flags & ex;
}

#define libc_feupdateenv_test  libc_feupdateenv_test_riscv
#define libc_feupdateenv_testf libc_feupdateenv_test_riscv
#define libc_feupdateenv_testl libc_feupdateenv_test_riscv

static __always_inline void
libc_feupdateenv_riscv (const fenv_t *envp)
{
  _FPU_SETCW (*envp | riscv_getflags ());
}

#define libc_feupdateenv  libc_feupdateenv_riscv
#define libc_feupdateenvf libc_feupdateenv_riscv
#define libc_feupdateenvl libc_feupdateenv_riscv

static __always_inline void
libc_feholdsetround_riscv (fenv_t *envp, int round)
{
  /* Note this implementation makes an improperly-formatted fenv_t and
     so should only be used in conjunction with libc_feresetround.  */
  int old_round;
  asm volatile ("csrrw %0, frm, %z1" : "=r" (old_round) : "rJ" (round));
  *envp = old_round;
}

#define libc_feholdsetround  libc_feholdsetround_riscv
#define libc_feholdsetroundf libc_feholdsetround_riscv
#define libc_feholdsetroundl libc_feholdsetround_riscv

static __always_inline void
libc_feresetround_riscv (fenv_t *envp)
{
  /* Note this implementation takes an improperly-formatted fenv_t and
     so should only be used in conjunction with libc_feholdsetround.  */
  riscv_setround (*envp);
}

#define libc_feresetround  libc_feresetround_riscv
#define libc_feresetroundf libc_feresetround_riscv
#define libc_feresetroundl libc_feresetround_riscv

#include_next <fenv_private.h>

#endif
