/* Macro for explicit loop unrolling.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef _LOOP_UNROLL_H
#define _LOOP_UNROLL_H

/* Loop unroll macro to be used for explicit force loop unrolling with a
   configurable number of iterations.  The idea is to make the loop unrolling
   independent of whether the compiler is able to unroll through specific
   optimizations options (-funroll-loops or -funroll-all-loops).

   For instance, to implement strcpy with SRC being the source input and
   DEST the destination buffer, it is expected the macro to be used in this
   way:

     #define ITERATION(index)	\
       ({ char c = *str++; *dest++ = c; c != '\0' })

     while (1)
       UNROLL_REPEAT (4, ITERATION)

   The loop will be manually unrolled 4 times.  Another option is to do
   the index update after the tests:

     #define ITERATION(index)	\
       ({ char c = *(str + index); *(dest + index) = c; c != '\0' })
     #define UPDATE(n)		\
       str += n; dst += n

     while (1)
       UNROLL_REPEAT_UPDATE (4, ITERATION, UPDATE)

   The loop will be manually unrolled 4 times and the SRC and DEST pointers
   will be updated only after the last iteration.

   Currently, both macros unroll the loop 8 times at maximum.  */

#define UNROLL_REPEAT_1(X)    if (!X(0)) break;
#define UNROLL_REPEAT_2(X)    UNROLL_REPEAT_1 (X) if (!X (1)) break;
#define UNROLL_REPEAT_3(X)    UNROLL_REPEAT_2 (X) if (!X (2)) break;
#define UNROLL_REPEAT_4(X)    UNROLL_REPEAT_3 (X) if (!X (3)) break;
#define UNROLL_REPEAT_5(X)    UNROLL_REPEAT_4 (X) if (!X (4)) break;
#define UNROLL_REPEAT_6(X)    UNROLL_REPEAT_5 (X) if (!X (5)) break;
#define UNROLL_REPEAT_7(X)    UNROLL_REPEAT_6 (X) if (!X (6)) break;
#define UNROLL_REPEAT_8(X)    UNROLL_REPEAT_7 (X) if (!X (7)) break;

#define UNROLL_EXPAND(...)    __VA_ARGS__

#define UNROLL_REPEAT__(N, X) UNROLL_EXPAND(UNROLL_REPEAT_ ## N) (X)
#define UNROLL_REPEAT_(N, X)  UNROLL_REPEAT__ (N, X)

#define UNROLL_REPEAT(N, X)                \
  (void) ({                                \
    UNROLL_REPEAT_ (UNROLL_EXPAND(N), X);  \
  })

#define UNROLL_REPEAT_UPDATE(N, X, U)      \
  (void) ({                                \
    UNROLL_REPEAT_ (UNROLL_EXPAND(N), X);  \
    UPDATE (N);                            \
  })

#endif
