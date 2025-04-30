/* Internal libc stuff for floating point environment routines.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#ifndef _FENV_LIBC_H
#define _FENV_LIBC_H	1

#include <fenv.h>
#include <ldsodefs.h>
#include <sysdep.h>

extern const fenv_t *__fe_nomask_env_priv (void);

extern const fenv_t *__fe_mask_env (void) attribute_hidden;

/* If the old env had any enabled exceptions and the new env has no enabled
   exceptions, then mask SIGFPE in the MSR FE0/FE1 bits.  This may allow the
   FPU to run faster because it always takes the default action and can not
   generate SIGFPE.  */
#define __TEST_AND_ENTER_NON_STOP(old, new) \
  do { \
    if (((old) & FPSCR_ENABLES_MASK) != 0 && ((new) & FPSCR_ENABLES_MASK) == 0) \
      (void) __fe_mask_env (); \
  } while (0)

/* If the old env has no enabled exceptions and the new env has any enabled
   exceptions, then unmask SIGFPE in the MSR FE0/FE1 bits.  This will put the
   hardware into "precise mode" and may cause the FPU to run slower on some
   hardware.  */
#define __TEST_AND_EXIT_NON_STOP(old, new) \
  do { \
    if (((old) & FPSCR_ENABLES_MASK) == 0 && ((new) & FPSCR_ENABLES_MASK) != 0) \
      (void) __fe_nomask_env_priv (); \
  } while (0)

/* The sticky bits in the FPSCR indicating exceptions have occurred.  */
#define FPSCR_STICKY_BITS ((FE_ALL_EXCEPT | FE_ALL_INVALID) & ~FE_INVALID)

/* Equivalent to fegetenv, but returns a fenv_t instead of taking a
   pointer.  */
#define fegetenv_register() __builtin_mffs()

/* Equivalent to fegetenv_register, but only returns bits for
   status, exception enables, and mode.
   Nicely, it turns out that the 'mffsl' instruction will decode to
   'mffs' on architectures older than "power9" because the additional
   bits set for 'mffsl' are "don't care" for 'mffs'.  'mffs' is a superset
   of 'mffsl'.  */
#define fegetenv_control()					\
  ({register double __fr;						\
    __asm__ __volatile__ (						\
      ".machine push; .machine \"power9\"; mffsl %0; .machine pop"	\
      : "=f" (__fr));							\
    __fr;								\
  })

#define __fe_mffscrn(rn)						\
  ({register fenv_union_t __fr;						\
    if (__builtin_constant_p (rn))					\
      __asm__ __volatile__ (						\
        ".machine push; .machine \"power9\"; mffscrni %0,%1; .machine pop" \
        : "=f" (__fr.fenv) : "i" (rn));					\
    else								\
    {									\
      __fr.l = (rn);							\
      __asm__ __volatile__ (						\
        ".machine push; .machine \"power9\"; mffscrn %0,%1; .machine pop" \
        : "=f" (__fr.fenv) : "f" (__fr.fenv));				\
    }									\
    __fr.fenv;								\
  })

/* Like fegetenv_control, but also sets the rounding mode.  */
#ifdef _ARCH_PWR9
#define fegetenv_and_set_rn(rn) __fe_mffscrn (rn)
#else
/* 'mffscrn' will decode to 'mffs' on ARCH < 3_00, which is still necessary
   but not sufficient, because it does not set the rounding mode.
   Explicitly set the rounding mode when 'mffscrn' actually doesn't.  */
#define fegetenv_and_set_rn(rn)						\
  ({register fenv_union_t __fr;						\
    __fr.fenv = __fe_mffscrn (rn);					\
    if (__glibc_unlikely (!(GLRO(dl_hwcap2) & PPC_FEATURE2_ARCH_3_00)))	\
      __fesetround_inline (rn);						\
    __fr.fenv;								\
  })
#endif

/* Equivalent to fesetenv, but takes a fenv_t instead of a pointer.  */
#define fesetenv_register(env) \
	do { \
	  double d = (env); \
	  if(GLRO(dl_hwcap) & PPC_FEATURE_HAS_DFP) \
	    asm volatile (".machine push; " \
			  ".machine \"power6\"; " \
			  "mtfsf 0xff,%0,1,0; " \
			  ".machine pop" : : "f" (d)); \
	  else \
	    __builtin_mtfsf (0xff, d); \
	} while(0)

/* Set the last 2 nibbles of the FPSCR, which contain the
   exception enables and the rounding mode.
   'fegetenv_control' retrieves these bits by reading the FPSCR.  */
#define fesetenv_control(env) __builtin_mtfsf (0b00000011, (env));

/* This very handy macro:
   - Sets the rounding mode to 'round to nearest';
   - Sets the processor into IEEE mode; and
   - Prevents exceptions from being raised for inexact results.
   These things happen to be exactly what you need for typical elementary
   functions.  */
#define relax_fenv_state() \
	do { \
	   if (GLRO(dl_hwcap) & PPC_FEATURE_HAS_DFP) \
	     asm volatile (".machine push; .machine \"power6\"; " \
		  "mtfsfi 7,0,1; .machine pop"); \
	   asm volatile ("mtfsfi 7,0"); \
	} while(0)

/* Set/clear a particular FPSCR bit (for instance,
   reset_fpscr_bit(FPSCR_VE);
   prevents INVALID exceptions from being raised).  */
#define set_fpscr_bit(x) asm volatile ("mtfsb1 %0" : : "i"(x))
#define reset_fpscr_bit(x) asm volatile ("mtfsb0 %0" : : "i"(x))

typedef union
{
  fenv_t fenv;
  unsigned long long l;
} fenv_union_t;


static inline int
__fesetround_inline (int round)
{
#ifdef _ARCH_PWR9
  __fe_mffscrn (round);
#else
  if (__glibc_likely (GLRO(dl_hwcap2) & PPC_FEATURE2_ARCH_3_00))
    __fe_mffscrn (round);
  else if ((unsigned int) round < 2)
    {
       asm volatile ("mtfsb0 30");
       if ((unsigned int) round == 0)
         asm volatile ("mtfsb0 31");
       else
         asm volatile ("mtfsb1 31");
    }
  else
    {
       asm volatile ("mtfsb1 30");
       if ((unsigned int) round == 2)
         asm volatile ("mtfsb0 31");
       else
         asm volatile ("mtfsb1 31");
    }
#endif
  return 0;
}

/* Same as __fesetround_inline, however without runtime check to use DFP
   mtfsfi syntax (as relax_fenv_state) or if round value is valid.  */
static inline void
__fesetround_inline_nocheck (const int round)
{
#ifdef _ARCH_PWR9
  __fe_mffscrn (round);
#else
  if (__glibc_likely (GLRO(dl_hwcap2) & PPC_FEATURE2_ARCH_3_00))
    __fe_mffscrn (round);
  else
    asm volatile ("mtfsfi 7,%0" : : "i" (round));
#endif
}

#define FPSCR_MASK(bit) (1 << (31 - (bit)))

/* Definitions of all the FPSCR bit numbers */
enum {
  FPSCR_FX = 0,    /* exception summary */
#define FPSCR_FX_MASK (FPSCR_MASK (FPSCR_FX))
  FPSCR_FEX,       /* enabled exception summary */
#define FPSCR_FEX_MASK (FPSCR_MASK FPSCR_FEX))
  FPSCR_VX,        /* invalid operation summary */
#define FPSCR_VX_MASK (FPSCR_MASK (FPSCR_VX))
  FPSCR_OX,        /* overflow */
#define FPSCR_OX_MASK (FPSCR_MASK (FPSCR_OX))
  FPSCR_UX,        /* underflow */
#define FPSCR_UX_MASK (FPSCR_MASK (FPSCR_UX))
  FPSCR_ZX,        /* zero divide */
#define FPSCR_ZX_MASK (FPSCR_MASK (FPSCR_ZX))
  FPSCR_XX,        /* inexact */
#define FPSCR_XX_MASK (FPSCR_MASK (FPSCR_XX))
  FPSCR_VXSNAN,    /* invalid operation for sNaN */
#define FPSCR_VXSNAN_MASK (FPSCR_MASK (FPSCR_VXSNAN))
  FPSCR_VXISI,     /* invalid operation for Inf-Inf */
#define FPSCR_VXISI_MASK (FPSCR_MASK (FPSCR_VXISI))
  FPSCR_VXIDI,     /* invalid operation for Inf/Inf */
#define FPSCR_VXIDI_MASK (FPSCR_MASK (FPSCR_VXIDI))
  FPSCR_VXZDZ,     /* invalid operation for 0/0 */
#define FPSCR_VXZDZ_MASK (FPSCR_MASK (FPSCR_VXZDZ))
  FPSCR_VXIMZ,     /* invalid operation for Inf*0 */
#define FPSCR_VXIMZ_MASK (FPSCR_MASK (FPSCR_VXIMZ))
  FPSCR_VXVC,      /* invalid operation for invalid compare */
#define FPSCR_VXVC_MASK (FPSCR_MASK (FPSCR_VXVC))
  FPSCR_FR,        /* fraction rounded [fraction was incremented by round] */
#define FPSCR_FR_MASK (FPSCR_MASK (FPSCR_FR))
  FPSCR_FI,        /* fraction inexact */
#define FPSCR_FI_MASK (FPSCR_MASK (FPSCR_FI))
  FPSCR_FPRF_C,    /* result class descriptor */
#define FPSCR_FPRF_C_MASK (FPSCR_MASK (FPSCR_FPRF_C))
  FPSCR_FPRF_FL,   /* result less than (usually, less than 0) */
#define FPSCR_FPRF_FL_MASK (FPSCR_MASK (FPSCR_FPRF_FL))
  FPSCR_FPRF_FG,   /* result greater than */
#define FPSCR_FPRF_FG_MASK (FPSCR_MASK (FPSCR_FPRF_FG))
  FPSCR_FPRF_FE,   /* result equal to */
#define FPSCR_FPRF_FE_MASK (FPSCR_MASK (FPSCR_FPRF_FE))
  FPSCR_FPRF_FU,   /* result unordered */
#define FPSCR_FPRF_FU_MASK (FPSCR_MASK (FPSCR_FPRF_FU))
  FPSCR_20,        /* reserved */
  FPSCR_VXSOFT,    /* invalid operation set by software */
#define FPSCR_VXSOFT_MASK (FPSCR_MASK (FPSCR_VXSOFT))
  FPSCR_VXSQRT,    /* invalid operation for square root */
#define FPSCR_VXSQRT_MASK (FPSCR_MASK (FPSCR_VXSQRT))
  FPSCR_VXCVI,     /* invalid operation for invalid integer convert */
#define FPSCR_VXCVI_MASK (FPSCR_MASK (FPSCR_VXCVI))
  FPSCR_VE,        /* invalid operation exception enable */
#define FPSCR_VE_MASK (FPSCR_MASK (FPSCR_VE))
  FPSCR_OE,        /* overflow exception enable */
#define FPSCR_OE_MASK (FPSCR_MASK (FPSCR_OE))
  FPSCR_UE,        /* underflow exception enable */
#define FPSCR_UE_MASK (FPSCR_MASK (FPSCR_UE))
  FPSCR_ZE,        /* zero divide exception enable */
#define FPSCR_ZE_MASK (FPSCR_MASK (FPSCR_ZE))
  FPSCR_XE,        /* inexact exception enable */
#define FPSCR_XE_MASK (FPSCR_MASK (FPSCR_XE))
#ifdef _ARCH_PWR6
  FPSCR_29,        /* Reserved in ISA 2.05  */
#define FPSCR_NI_MASK (FPSCR_MASK (FPSCR_29))
#else
  FPSCR_NI,        /* non-IEEE mode (typically, no denormalised numbers) */
#define FPSCR_NI_MASK (FPSCR_MASK (FPSCR_NI))
#endif /* _ARCH_PWR6 */
  /* the remaining two least-significant bits keep the rounding mode */
  FPSCR_RN_hi,
#define FPSCR_RN_hi_MASK (FPSCR_MASK (FPSCR_RN_hi))
  FPSCR_RN_lo
#define FPSCR_RN_lo_MASK (FPSCR_MASK (FPSCR_RN_lo))
};

#define FPSCR_RN_MASK (FPSCR_RN_hi_MASK|FPSCR_RN_lo_MASK)
#define FPSCR_ENABLES_MASK \
  (FPSCR_VE_MASK|FPSCR_OE_MASK|FPSCR_UE_MASK|FPSCR_ZE_MASK|FPSCR_XE_MASK)
#define FPSCR_BASIC_EXCEPTIONS_MASK \
  (FPSCR_VX_MASK|FPSCR_OX_MASK|FPSCR_UX_MASK|FPSCR_ZX_MASK|FPSCR_XX_MASK)
#define FPSCR_EXCEPTIONS_MASK (FPSCR_BASIC_EXCEPTIONS_MASK| \
  FPSCR_VXSNAN_MASK|FPSCR_VXISI_MASK|FPSCR_VXIDI_MASK|FPSCR_VXZDZ_MASK| \
  FPSCR_VXIMZ_MASK|FPSCR_VXVC_MASK|FPSCR_VXSOFT_MASK|FPSCR_VXSQRT_MASK| \
  FPSCR_VXCVI_MASK)
#define FPSCR_FPRF_MASK \
  (FPSCR_FPRF_C_MASK|FPSCR_FPRF_FL_MASK|FPSCR_FPRF_FG_MASK| \
   FPSCR_FPRF_FE_MASK|FPSCR_FPRF_FU_MASK)
#define FPSCR_CONTROL_MASK (FPSCR_ENABLES_MASK|FPSCR_NI_MASK|FPSCR_RN_MASK)
#define FPSCR_STATUS_MASK (FPSCR_FR_MASK|FPSCR_FI_MASK|FPSCR_FPRF_MASK)

/* The bits in the FENV(1) ABI for exceptions correspond one-to-one with bits
   in the FPSCR, albeit shifted to different but corresponding locations.
   Similarly, the exception indicator bits in the FPSCR correspond one-to-one
   with the exception enable bits. It is thus possible to map the FENV(1)
   exceptions directly to the FPSCR enables with a simple mask and shift,
   and vice versa. */
#define FPSCR_EXCEPT_TO_ENABLE_SHIFT 22

static inline int
fenv_reg_to_exceptions (unsigned long long l)
{
  return (((int)l) & FPSCR_ENABLES_MASK) << FPSCR_EXCEPT_TO_ENABLE_SHIFT;
}

static inline unsigned long long
fenv_exceptions_to_reg (int excepts)
{
  return (unsigned long long)
    (excepts & FE_ALL_EXCEPT) >> FPSCR_EXCEPT_TO_ENABLE_SHIFT;
}

#ifdef _ARCH_PWR6
  /* Not supported in ISA 2.05.  Provided for source compat only.  */
# define FPSCR_NI 29
#endif /* _ARCH_PWR6 */

/* This operation (i) sets the appropriate FPSCR bits for its
   parameter, (ii) converts sNaN to the corresponding qNaN, and (iii)
   otherwise passes its parameter through unchanged (in particular, -0
   and +0 stay as they were).  The `obvious' way to do this is optimised
   out by gcc.  */
#define f_wash(x) \
   ({ double d; asm volatile ("fmul %0,%1,%2" \
			      : "=f"(d) \
			      : "f" (x), "f"((float)1.0)); d; })
#define f_washf(x) \
   ({ float f; asm volatile ("fmuls %0,%1,%2" \
			     : "=f"(f) \
			     : "f" (x), "f"((float)1.0)); f; })

#endif /* fenv_libc.h */
