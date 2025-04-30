/* Internal header file for <setjmp.h>.  Linux/x86 version.
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

#ifndef	_SETJMPP_H
#define	_SETJMPP_H	1

#include <bits/types/__sigset_t.h>
#include <libc-pointer-arith.h>
#include <sigsetops.h>

/* <setjmp/setjmp.h> has

struct __jmp_buf_tag
  {
    __jmp_buf __jmpbuf;
    int __mask_was_saved;
    __sigset_t __saved_mask;
  };

   struct __jmp_buf_tag is 32 bits aligned on i386 and is 64 bits
   aligned on x32 and x86-64.  __saved_mask is aligned to 32 bits
   on i386/x32 without padding and is aligned to 64 bits on x86-64
   with 32 bit padding.

   and <nptl/descr.h> has

struct pthread_unwind_buf
{
  struct
  {
    __jmp_buf jmp_buf;
    int mask_was_saved;
  } cancel_jmp_buf[1];

  union
  {
    void *pad[4];
    struct
    {
      struct pthread_unwind_buf *prev;
      struct _pthread_cleanup_buffer *cleanup;
      int canceltype;
    } data;
  } priv;
};

   struct pthread_unwind_buf is 32 bits aligned on i386 and 64 bits
   aligned on x32/x86-64.  cancel_jmp_buf is aligned to 32 bits on
   i386 and is aligned to 64 bits on x32/x86-64.

   The pad array in struct pthread_unwind_buf is used by setjmp to save
   shadow stack register.  The usable space in __saved_mask for sigset
   and shadow stack pointer:
   1. i386: The 4x4 byte pad array which can be used for 4 byte shadow
   stack pointer and maximum 12 byte sigset.
   2. x32: 4 byte padding + the 4x4 byte pad array which can be used
   for 8 byte shadow stack pointer and maximum 12 byte sigset.
   3. x86-64: The 4x8 byte pad array which can be used for 8 byte
   shadow stack pointer and maximum 24 byte sigset.

   NB: We use setjmp in thread cancellation and this saves the shadow
   stack register, but __libc_unwind_longjmp doesn't restore the shadow
   stack register since cancellation never returns after longjmp.  */

/* Number of bits per long.  */
#define _JUMP_BUF_SIGSET_BITS_PER_WORD (8 * sizeof (unsigned long int))
/* The biggest signal number.  As of kernel 4.14, x86 _NSIG is 64. The
   common maximum sigset for i386, x32 and x86-64 is 12 bytes (96 bits).
   Define it to 96 to leave some rooms for future use.  */
#define _JUMP_BUF_SIGSET_NSIG	96
/* Number of longs to hold all signals.  */
#define _JUMP_BUF_SIGSET_NWORDS \
  (ALIGN_UP (_JUMP_BUF_SIGSET_NSIG, _JUMP_BUF_SIGSET_BITS_PER_WORD) \
   / _JUMP_BUF_SIGSET_BITS_PER_WORD)

typedef struct
  {
    unsigned long int __val[_JUMP_BUF_SIGSET_NWORDS];
  } __jmp_buf_sigset_t;

typedef union
  {
    __sigset_t __saved_mask_compat;
    struct
      {
	__jmp_buf_sigset_t __saved_mask;
	/* Used for shadow stack pointer.  NB: Shadow stack pointer
	   must have the same alignment as __saved_mask.  Otherwise
	   offset of __saved_mask will be changed.  */
	unsigned long int __shadow_stack_pointer;
      } __saved;
  } __jmpbuf_arch_t;

#undef __sigset_t
#define __sigset_t __jmpbuf_arch_t
#include <setjmp.h>
#undef __saved_mask
#define __saved_mask __saved_mask.__saved.__saved_mask

#include <signal.h>

typedef struct
  {
    unsigned long int __val[__NSIG_WORDS];
  } __sigprocmask_sigset_t;

extern jmp_buf ___buf;
extern  __typeof (___buf[0].__saved_mask) ___saved_mask;
_Static_assert (sizeof (___saved_mask) >= sizeof (__sigprocmask_sigset_t),
		"size of ___saved_mask < size of __sigprocmask_sigset_t");

#endif /* setjmpP.h  */
