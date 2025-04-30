/* Dynamic loading of the libgcc unwinder.  MIPS customization.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _ARCH_UNWIND_LINK_H
#define _ARCH_UNWIND_LINK_H

#include <stdint.h>
#include <sys/syscall.h>

#define UNWIND_LINK_GETIP 1
#define UNWIND_LINK_FRAME_STATE_FOR 1
#define UNWIND_LINK_FRAME_ADJUSTMENT 1
#define UNWIND_LINK_EXTRA_FIELDS
#define UNWIND_LINK_EXTRA_INIT

/* MIPS fallback code handle a frame where its FDE can not be obtained
   (for instance a signal frame) by reading the kernel allocated signal frame
   and adding '2' to the value of 'sc_pc' [1].  The added value is used to
   recognize an end of an EH region on mips16 [2].

   The idea here is to adjust the obtained signal frame ADDR value and remove
   the libgcc added value by checking if the previous frame is a signal frame
   one.

   [1] libgcc/config/mips/linux-unwind.h from gcc code.
   [2] gcc/config/mips/mips.h from gcc code.  */

static inline void *
unwind_arch_adjustment (void *prev, void *addr)
{
  uint32_t *pc = (uint32_t *) prev;

  if (pc == NULL)
    return addr;

  /* For MIPS16 or microMIPS frame libgcc makes no adjustment.  */
  if ((uintptr_t) pc & 0x3)
    return addr;

  /* The vDSO containes either

     24021061 li v0, 0x1061 (rt_sigreturn)
     0000000c syscall
        or
     24021017 li v0, 0x1017 (sigreturn)
     0000000c syscall  */
  if (pc[1] != 0x0000000c)
    return addr;
#if _MIPS_SIM == _ABIO32
  if (pc[0] == (0x24020000 | __NR_sigreturn))
    return (void *) ((uintptr_t) addr - 2);
#endif
  if (pc[0] == (0x24020000 | __NR_rt_sigreturn))
    return (void *) ((uintptr_t) addr - 2);

  return addr;
}

#endif /* _ARCH_UNWIND_LINK_H */
