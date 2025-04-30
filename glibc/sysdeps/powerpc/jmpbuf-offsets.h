/* Private macros for accessing __jmp_buf contents.  PowerPC version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#define JB_GPR1   0  /* Also known as the stack pointer */
#define JB_GPR2   1
#define JB_LR     2  /* The address we will return to */
#if __WORDSIZE == 64
# define JB_GPRS   3  /* GPRs 14 through 31 are saved, 18*2 words total.  */
# define JB_CR     21 /* Shared dword with VRSAVE.  CR word at offset 172.  */
# define JB_FPRS   22 /* FPRs 14 through 31 are saved, 18*2 words total.  */
# define JB_SIZE   (64 * 8) /* As per PPC64-VMX ABI.  */
# define JB_VRSAVE 21 /* Shared dword with CR.  VRSAVE word at offset 168.  */
# define JB_VRS    40 /* VRs 20 through 31 are saved, 12*4 words total.  */
#else
# define JB_GPRS   3  /* GPRs 14 through 31 are saved, 18 in total.  */
# define JB_CR     21 /* Condition code registers.  */
# define JB_FPRS   22 /* FPRs 14 through 31 are saved, 18*2 words total.  */
# define JB_SIZE   ((64 + (12 * 4)) * 4)
# define JB_VRSAVE 62
# define JB_VRS    64
#endif
