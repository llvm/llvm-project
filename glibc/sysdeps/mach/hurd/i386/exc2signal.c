/* Translate Mach exception codes into signal numbers.  i386 version.
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

#include <hurd.h>
#include <hurd/signal.h>
#include <mach/exception.h>

/* Translate the Mach exception codes, as received in an `exception_raise' RPC,
   into a signal number and signal subcode.  */

static void
exception2signal (struct hurd_signal_detail *detail, int *signo, int posix)
{
  detail->error = 0;

  switch (detail->exc)
    {
    default:
      *signo = SIGIOT;
      detail->code = detail->exc;
      break;

    case EXC_BAD_ACCESS:
      switch (detail->exc_code)
        {
	case KERN_INVALID_ADDRESS:
	case KERN_MEMORY_FAILURE:
	  *signo = SIGSEGV;
	  detail->code = posix ? SEGV_MAPERR : detail->exc_subcode;
	  break;

	case KERN_PROTECTION_FAILURE:
	case KERN_WRITE_PROTECTION_FAILURE:
	  *signo = SIGSEGV;
	  detail->code = posix ? SEGV_ACCERR : detail->exc_subcode;
	  break;

	default:
	  *signo = SIGBUS;
	  detail->code = posix ? BUS_ADRERR : detail->exc_subcode;
	  break;
	}
      detail->error = detail->exc_code;
      break;

    case EXC_BAD_INSTRUCTION:
      *signo = SIGILL;
      switch (detail->exc_code)
        {
	case EXC_I386_INVOP:
	  detail->code = posix ? ILL_ILLOPC : ILL_INVOPR_FAULT;
	  break;

	case EXC_I386_STKFLT:
	  detail->code = posix ? ILL_BADSTK : ILL_STACK_FAULT;
	  break;

	default:
	  detail->code = 0;
	  break;
	}
      break;

    case EXC_ARITHMETIC:
      *signo = SIGFPE;
      switch (detail->exc_code)
	{
	case EXC_I386_DIV:	/* integer divide by zero */
	  detail->code = posix ? FPE_INTDIV : FPE_INTDIV_FAULT;
	  break;

	case EXC_I386_INTO:	/* integer overflow */
	  detail->code = posix ? FPE_INTOVF : FPE_INTOVF_TRAP;
	  break;

	  /* These aren't anywhere documented or used in Mach 3.0.  */
	case EXC_I386_NOEXT:
	case EXC_I386_EXTOVR:
	default:
	  detail->code = 0;
	  break;

	case EXC_I386_EXTERR:
	  /* Subcode is the fp_status word saved by the hardware.
	     Give an error code corresponding to the first bit set.  */
	  if (detail->exc_subcode & FPS_IE)
	    {
	      /* NB: We used to send SIGILL here but we can't distinguish
		 POSIX vs. legacy with respect to what signal we send.  */
	      detail->code = posix ? FPE_FLTINV : 0 /*ILL_FPEOPR_FAULT*/;
	    }
	  else if (detail->exc_subcode & FPS_DE)
	    {
	      detail->code = posix ? FPE_FLTUND : FPE_FLTDNR_FAULT;
	    }
	  else if (detail->exc_subcode & FPS_ZE)
	    {
	      detail->code = posix ? FPE_FLTDIV : FPE_FLTDIV_FAULT;
	    }
	  else if (detail->exc_subcode & FPS_OE)
	    {
	      detail->code = posix ? FPE_FLTOVF : FPE_FLTOVF_FAULT;
	    }
	  else if (detail->exc_subcode & FPS_UE)
	    {
	      detail->code = posix ? FPE_FLTUND : FPE_FLTUND_FAULT;
	    }
	  else if (detail->exc_subcode & FPS_PE)
	    {
	      detail->code = posix ? FPE_FLTRES : FPE_FLTINX_FAULT;
	    }
	  else
	    {
	      detail->code = 0;
	    }
	  break;

	  /* These two can only be arithmetic exceptions if we
	     are in V86 mode.  (See Mach 3.0 i386/trap.c.)  */
	case EXC_I386_EMERR:
	  detail->code = posix ? 0 : FPE_EMERR_FAULT;
	  break;
	case EXC_I386_BOUND:
	  detail->code = posix ? FPE_FLTSUB : FPE_EMBND_FAULT;
	  break;
	}
      break;

    case EXC_EMULATION:
      /* 3.0 doesn't give this one, why, I don't know.  */
      *signo = SIGEMT;
      detail->code = 0;
      break;

    case EXC_SOFTWARE:
      /* The only time we get this in Mach 3.0
	 is for an out of bounds trap.  */
      if (detail->exc_code == EXC_I386_BOUND)
	{
	  *signo = SIGFPE;
	  detail->code = posix ? FPE_FLTSUB : FPE_SUBRNG_FAULT;
	}
      else
	{
	  *signo = SIGEMT;
	  detail->code = 0;
	}
      break;

    case EXC_BREAKPOINT:
      *signo = SIGTRAP;
      switch (detail->exc_code)
        {
	case EXC_I386_SGL:
	  detail->code = posix ? TRAP_BRKPT : DBG_SINGLE_TRAP;
	  break;

	case EXC_I386_BPT:
	  detail->code = posix ? TRAP_BRKPT : DBG_BRKPNT_FAULT;
	  break;

	default:
	  detail->code = 0;
	  break;
	}
      break;
    }
}
libc_hidden_def (_hurd_exception2signal)

void
_hurd_exception2signal (struct hurd_signal_detail *detail, int *signo)
{
  exception2signal (detail, signo, 1);
}

void
_hurd_exception2signal_legacy (struct hurd_signal_detail *detail, int *signo)
{
  exception2signal (detail, signo, 0);
}
