/* Machine-dependent signal context structure for GNU Hurd.  i386 version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef _BITS_SIGCONTEXT_H
#define _BITS_SIGCONTEXT_H 1

#if !defined _SIGNAL_H && !defined _SYS_UCONTEXT_H
# error "Never use <bits/sigcontext.h> directly; include <signal.h> instead."
#endif

/* Signal handlers are actually called:
   void handler (int sig, int code, struct sigcontext *scp);  */

#include <bits/types/__sigset_t.h>
#include <mach/machine/fp_reg.h>

/* State of this thread when the signal was taken.  */
struct sigcontext
  {
    /* These first members are machine-independent.  */

    int sc_onstack;		/* Nonzero if running on sigstack.  */
    __sigset_t sc_mask;		/* Blocked signals to restore.  */

    /* MiG reply port this thread is using.  */
    unsigned int sc_reply_port;

    /* Port this thread is doing an interruptible RPC on.  */
    unsigned int sc_intr_port;

    /* Error code associated with this signal (interpreted as `error_t').  */
    int sc_error;

    /* All following members are machine-dependent.  The rest of this
       structure is written to be laid out identically to:
       {
	 struct i386_thread_state basic;
	 struct i386_float_state fpu;
       }
       trampoline.c knows this, so it must be changed if this changes.  */

#define sc_i386_thread_state sc_gs /* Beginning of correspondence.  */
    /* Segment registers.  */
    int sc_gs;
    int sc_fs;
    int sc_es;
    int sc_ds;

    /* "General" registers.  These members are in the order that the i386
       `pusha' and `popa' instructions use (`popa' ignores %esp).  */
    int sc_edi;
    int sc_esi;
    int sc_ebp;
    int sc_esp;			/* Not used; sc_uesp is used instead.  */
    int sc_ebx;
    int sc_edx;
    int sc_ecx;
    int sc_eax;

    int sc_eip;			/* Instruction pointer.  */
    int sc_cs;			/* Code segment register.  */

    int sc_efl;			/* Processor flags.  */

    int sc_uesp;		/* This stack pointer is used.  */
    int sc_ss;			/* Stack segment register.  */

    /* Following mimics struct i386_float_state.  Structures and symbolic
       values can be found in <mach/i386/fp_reg.h>.  */
#define sc_i386_float_state sc_fpkind
    int sc_fpkind;		/* FP_NO, FP_387, etc.  */
    int sc_fpused;		/* If zero, ignore rest of float state.  */
    struct i386_fp_save sc_fpsave;
    struct i386_fp_regs sc_fpregs;
    int sc_fpexcsr;		/* FPSR including exception bits.  */
  };

/* Traditional BSD names for some members.  */
#define sc_sp	sc_uesp		/* Stack pointer.  */
#define sc_fp	sc_ebp		/* Frame pointer.  */
#define sc_pc	sc_eip		/* Process counter.  */
#define sc_ps	sc_efl


/* The deprecated sigcode values below are passed as an extra, non-portable
   argument to regular signal handlers.  You should use SA_SIGINFO handlers
   instead, which use the standard POSIX signal codes.  */

/* Codes for SIGFPE.  */
#define FPE_INTOVF_TRAP		0x1 /* integer overflow */
#define FPE_INTDIV_FAULT	0x2 /* integer divide by zero */
#define FPE_FLTOVF_FAULT	0x3 /* floating overflow */
#define FPE_FLTDIV_FAULT	0x4 /* floating divide by zero */
#define FPE_FLTUND_FAULT	0x5 /* floating underflow */
#define FPE_SUBRNG_FAULT	0x7 /* BOUNDS instruction failed */
#define FPE_FLTDNR_FAULT	0x8 /* denormalized operand */
#define FPE_FLTINX_FAULT	0x9 /* floating loss of precision */
#define FPE_EMERR_FAULT		0xa /* mysterious emulation error 33 */
#define FPE_EMBND_FAULT		0xb /* emulation BOUNDS instruction failed */

/* Codes for SIGILL.  */
#define ILL_INVOPR_FAULT	0x1 /* invalid operation */
#define ILL_STACK_FAULT		0x2 /* fault on microkernel stack access */
#define ILL_FPEOPR_FAULT	0x3 /* invalid floating operation */

/* Codes for SIGTRAP.  */
#define DBG_SINGLE_TRAP		0x1 /* single step */
#define DBG_BRKPNT_FAULT	0x2 /* breakpoint instruction */

#endif /* bits/sigcontext.h */
