/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <stdarg.h>
#include <sysdep.h>
#include <errno.h>

/* HPPA implements syscall() in 'C'; the assembler version would
   typically be in syscall.S. Also note that we have INLINE_SYSCALL,
   INTERNAL_SYSCALL, and all the generated pure assembly syscall wrappers.
   How often the function is used is unknown. */

long int
syscall (long int __sysno, ...)
{
  /* FIXME: Keep this matching INLINE_SYSCALL for hppa */
  va_list args;
  long int arg0, arg1, arg2, arg3, arg4, arg5;
  long int __sys_res;

  /* Load varargs */
  va_start (args, __sysno);
  arg0 = va_arg (args, long int);
  arg1 = va_arg (args, long int);
  arg2 = va_arg (args, long int);
  arg3 = va_arg (args, long int);
  arg4 = va_arg (args, long int);
  arg5 = va_arg (args, long int);
  va_end (args);

  {
    LOAD_ARGS_6 (arg0, arg1, arg2, arg3, arg4, arg5)
    register unsigned long int __res asm("r28");
    PIC_REG_DEF
    LOAD_REGS_6
    asm volatile (SAVE_ASM_PIC
		  "	ble  0x100(%%sr2, %%r0)	\n"
		  "	copy %1, %%r20		\n"
		  LOAD_ASM_PIC
		  : "=r" (__res)
		  : "r" (__sysno) PIC_REG_USE ASM_ARGS_6
		  : "memory", CALL_CLOB_REGS CLOB_ARGS_6);
    __sys_res = __res;
  }
  if ((unsigned long int) __sys_res >= (unsigned long int) -4095)
    {
      __set_errno (-__sys_res);
      __sys_res = -1;
    }
  return __sys_res;
}
