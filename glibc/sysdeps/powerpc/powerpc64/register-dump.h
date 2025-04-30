/* Dump registers.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <sys/uio.h>
#include <_itoa.h>

/* This prints out the information in the following form: */
static const char dumpform[] = "\
Register dump:\n\
sr0=000000000000020% sr1=000000000000021% dar=000000000000029% dsi=000002a%\n\
lr=000000000000024%  ctr=000000000000023% gr3*=000000000000022% trap=0000028%\n\
ccr=0000026%  xer=0000025%\n\
gr0-3:   000000000000000% 000000000000001% 000000000000002% 000000000000003%\n\
gr4-7:   000000000000004% 000000000000005% 000000000000006% 000000000000007%\n\
gr8-11:  000000000000008% 000000000000009% 00000000000000a% 00000000000000b%\n\
gr12-15: 00000000000000c% 00000000000000d% 00000000000000e% 00000000000000f%\n\
gr16-19: 000000000000010% 000000000000011% 000000000000012% 000000000000013%\n\
gr20-23: 000000000000014% 000000000000015% 000000000000016% 000000000000017%\n\
gr24-27: 000000000000018% 000000000000019% 00000000000001a% 00000000000001b%\n\
gr28-31: 00000000000001c% 00000000000001d% 00000000000001e% 00000000000001f%\n\
fscr=000000000000050%\n\
fp0-3:   000000000000030% 000000000000031% 000000000000032% 000000000000033%\n\
fp4-7:   000000000000034% 000000000000035% 000000000000036% 000000000000037%\n\
fp8-11:  000000000000038% 000000000000038% 00000000000003a% 00000000000003b%\n\
fp12-15: 00000000000003c% 00000000000003d% 00000000000003e% 00000000000003f%\n\
fp16-19: 000000000000040% 000000000000041% 000000000000042% 000000000000043%\n\
fp20-23: 000000000000044% 000000000000045% 000000000000046% 000000000000047%\n\
fp24-27: 000000000000048% 000000000000049% 00000000000004a% 00000000000004b%\n\
fp28-31: 00000000000004c% 00000000000004d% 00000000000004e% 00000000000004f%\n\
";

/* Most of the fields are self-explanatory.  'sr0' is the next
   instruction to execute, from SRR0, which may have some relationship
   with the instruction that caused the exception.  'r3*' is the value
   that will be returned in register 3 when the current system call
   returns.  'sr1' is SRR1, bits 16-31 of which are copied from the MSR:

   16 - External interrupt enable
   17 - Privilege level (1=user, 0=supervisor)
   18 - FP available
   19 - Machine check enable (if clear, processor locks up on machine check)
   20 - FP exception mode bit 0 (FP exceptions recoverable)
   21 - Single-step trace enable
   22 - Branch trace enable
   23 - FP exception mode bit 1
   25 - exception prefix (if set, exceptions are taken from 0xFFFnnnnn,
        otherwise from 0x000nnnnn).
   26 - Instruction address translation enabled.
   27 - Data address translation enabled.
   30 - Exception is recoverable (otherwise, don't try to return).
   31 - Little-endian mode enable.

   'Trap' is the address of the exception:

   00200 - Machine check exception (memory parity error, for instance)
   00300 - Data access exception (memory not mapped, see dsisr for why)
   00400 - Instruction access exception (memory not mapped)
   00500 - External interrupt
   00600 - Alignment exception (see dsisr for more information)
   00700 - Program exception (illegal/trap instruction, FP exception)
   00800 - FP unavailable (should not be seen by user code)
   00900 - Decrementer exception (for instance, SIGALRM)
   00A00 - I/O controller interface exception
   00C00 - System call exception (for instance, kill(3)).
   00E00 - FP assist exception (optional FP instructions, etc.)

   'dar' is the memory location, for traps 00300, 00400, 00600, 00A00.
   'dsisr' has the following bits under trap 00300:
   0 - direct-store error exception
   1 - no page table entry for page
   4 - memory access not permitted
   5 - trying to access I/O controller space or using lwarx/stwcx on
       non-write-cached memory
   6 - access was store
   9 - data access breakpoint hit
   10 - segment table search failed to find translation (64-bit ppcs only)
   11 - I/O controller instruction not permitted
   For trap 00400, the same bits are set in SRR1 instead.
   For trap 00600, bits 12-31 of the DSISR set to allow emulation of
   the instruction without actually having to read it from memory.
*/

#define xtoi(x) (x >= 'a' ? x + 10 - 'a' : x - '0')

static void
register_dump (int fd, struct sigcontext *ctx)
{
  char buffer[sizeof (dumpform)];
  char *bufferpos;
  unsigned regno;
  unsigned long *regs = (unsigned long *)(ctx->regs);

  memcpy(buffer, dumpform, sizeof (dumpform));

  /* Generate the output.  */
  while ((bufferpos = memchr (buffer, '%', sizeof (dumpform))))
    {
      regno = xtoi (bufferpos[-1]) | xtoi (bufferpos[-2]) << 4;
      memset (bufferpos-2, '0', 3);
      _itoa_word (regs[regno], bufferpos+1, 16, 0);
    }

  /* Write the output.  */
  write (fd, buffer, sizeof (buffer) - 1);
}


#define REGISTER_DUMP \
  register_dump (fd, ctx)
