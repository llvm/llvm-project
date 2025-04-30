/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

/* Copied from setjmp/setjmp.h, powerpc/bits/setjmp.h and modified
   appropriately to keep backward compatible with setjmp without
   AltiVec/VMX support.

   This file is not exported and the interfaces are private to libc.  */

#ifndef	__NOVMX_SETJMP_H
#define	__NOVMX_SETJMP_H	1

#include <bits/wordsize.h>

/* The following definitions are needed by ASM implementations of the old
   (novmx) __longjmp/__setjmp functions.  */

# define JB_GPR1   0  /* Also known as the stack pointer */
# define JB_GPR2   1
# define JB_LR     2  /* The address we will return to */
# if __WORDSIZE == 64
#  define JB_GPRS   3  /* GPRs 14 through 31 are saved, 18*2 words total.  */
#  define JB_CR     21 /* Condition code registers. */
#  define JB_FPRS   22 /* FPRs 14 through 31 are saved, 18*2 words total.  */
#  define JB_SIZE   (40 * 8)
# else
#  define JB_GPRS   3  /* GPRs 14 through 31 are saved, 18 in total.  */
#  define JB_CR     21 /* Condition code registers.  */
#  define JB_FPRS   22 /* FPRs 14 through 31 are saved, 18*2 words total.  */
#  define JB_SIZE   (58 * 4)
# endif

#ifndef	_ASM
/* The following definitions are needed by the novmx* implementations of
   setjmp/longjmp/sigsetjmp/etc that wrapper __setjmp/__longjmp.  */

# if __WORDSIZE == 64
typedef long int __jmp_buf[40];
# else
typedef long int __jmp_buf[58];
# endif

# include <bits/types/__sigset_t.h>

/* Calling environment, plus possibly a saved signal mask.  */
typedef struct __novmx__jmp_buf_tag
  {
    /* NOTE: The machine-dependent definitions of `__sigsetjmp'
       assume that a `jmp_buf' begins with a `__jmp_buf' and that
       `__mask_was_saved' follows it.  Do not move these members
       or add others before it.  */
    __jmp_buf __jmpbuf;		/* Calling environment.  */
    int __mask_was_saved;	/* Saved the signal mask?  */
    __sigset_t __saved_mask;	/* Saved signal mask.  */
  } __novmx__jmp_buf[1];


/* Store the calling environment in ENV, also saving the signal mask.
   Return 0.  */
extern int __novmxsetjmp (__novmx__jmp_buf __env);

/* Store the calling environment in ENV, also saving the
   signal mask if SAVEMASK is nonzero.  Return 0.
   This is the internal name for `sigsetjmp'.  */
extern int __novmx__sigsetjmp (struct __novmx__jmp_buf_tag __env[1],
			       int __savemask);

/* Store the calling environment in ENV, not saving the signal mask.
   Return 0.  */
extern int __novmx_setjmp (struct __novmx__jmp_buf_tag __env[1]);

/* Jump to the environment saved in ENV, making the
   `setjmp' call there return VAL, or 1 if VAL is 0.  */
extern void __novmxlongjmp (struct __novmx__jmp_buf_tag __env[1], int __val)
     __attribute__ ((__noreturn__));

/* Same.  Usually `_longjmp' is used with `_setjmp', which does not save
   the signal mask.  But it is how ENV was saved that determines whether
   `longjmp' restores the mask; `_longjmp' is just an alias.  */
extern void __novmx_longjmp (struct __novmx__jmp_buf_tag __env[1], int __val)
     __attribute__ ((__noreturn__));

/* Use the same type for `jmp_buf' and `sigjmp_buf'.
   The `__mask_was_saved' flag determines whether
   or not `longjmp' will restore the signal mask.  */
typedef struct __novmx__jmp_buf_tag __novmx__sigjmp_buf[1];

/* Jump to the environment saved in ENV, making the
   sigsetjmp call there return VAL, or 1 if VAL is 0.
   Restore the signal mask if that sigsetjmp call saved it.
   This is just an alias `longjmp'.  */
extern void __novmxsiglongjmp (__novmx__sigjmp_buf __env, int __val)
     __attribute__ ((__noreturn__));

/* Internal machine-dependent function to restore context sans signal mask.  */
extern void __novmx__longjmp (__jmp_buf __env, int __val)
     __attribute__ ((__noreturn__));

/* Internal function to possibly save the current mask of blocked signals
   in ENV, and always set the flag saying whether or not it was saved.
   This is used by the machine-dependent definition of `__sigsetjmp'.
   Always returns zero, for convenience.  */
extern int __novmx__sigjmp_save (__novmx__jmp_buf __env, int __savemask);

extern void _longjmp_unwind (__novmx__jmp_buf env, int val);

extern void __novmx__libc_siglongjmp (__novmx__sigjmp_buf env, int val)
          __attribute__ ((noreturn));

extern void __novmx__libc_longjmp (__novmx__sigjmp_buf env, int val)
     __attribute__ ((noreturn));

libc_hidden_proto (__novmx__libc_longjmp)
libc_hidden_proto (__novmx_setjmp)
libc_hidden_proto (__novmx__sigsetjmp)
#endif /* !_ASM */

#endif
