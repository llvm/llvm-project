/* jump buffer constants for RISC-V
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Produced by this program:

   #include <stdio.h>
   #include <unistd.h>
   #include <setjmp.h>
   #include <stddef.h>

   int main (int argc, char **argv)
   {
       printf ("#define JMP_BUF_SIZE %d\n", sizeof (jmp_buf));
       printf ("#define JMP_BUF_ALIGN %d\n", __alignof__ (jmp_buf));
       printf ("#define SIGJMP_BUF_SIZE %d\n", sizeof (sigjmp_buf));
       printf ("#define SIGJMP_BUF_ALIGN %d\n", __alignof__ (sigjmp_buf));
       printf ("#define MASK_WAS_SAVED_OFFSET %d\n", offsetof (struct __jmp_buf_tag, __mask_was_saved));
       printf ("#define SAVED_MASK_OFFSET %d\n", offsetof (struct __jmp_buf_tag, __saved_mask));
   } */

#if defined __riscv_float_abi_soft
# define JMP_BUF_SIZE 248
# define JMP_BUF_ALIGN 8
# define SIGJMP_BUF_SIZE 248
# define SIGJMP_BUF_ALIGN 8
# define MASK_WAS_SAVED_OFFSET 112
# define SAVED_MASK_OFFSET 120
#elif defined __riscv_float_abi_double
# define JMP_BUF_SIZE 344
# define JMP_BUF_ALIGN 8
# define SIGJMP_BUF_SIZE 344
# define SIGJMP_BUF_ALIGN 8
# define MASK_WAS_SAVED_OFFSET 208
# define SAVED_MASK_OFFSET 216
#else
# error "Unknown RISC-V floating-point ABI"
#endif
