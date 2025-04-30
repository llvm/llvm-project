/* Copyright (C) 2003-2021 Free Software Foundation, Inc.

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

#include <stdint.h>
#include <sysdep.h>


typedef int8_t atomic8_t;
typedef uint8_t uatomic8_t;
typedef int_fast8_t atomic_fast8_t;
typedef uint_fast8_t uatomic_fast8_t;

typedef int32_t atomic32_t;
typedef uint32_t uatomic32_t;
typedef int_fast32_t atomic_fast32_t;
typedef uint_fast32_t uatomic_fast32_t;

typedef intptr_t atomicptr_t;
typedef uintptr_t uatomicptr_t;
typedef intmax_t atomic_max_t;
typedef uintmax_t uatomic_max_t;

#define __HAVE_64B_ATOMICS 0
#define USE_ATOMIC_COMPILER_BUILTINS 0

/* XXX Is this actually correct?  */
#define ATOMIC_EXCHANGE_USES_CAS 1


/* Microblaze does not have byte and halfword forms of load and reserve and
   store conditional. So for microblaze we stub out the 8- and 16-bit forms.  */
#define __arch_compare_and_exchange_bool_8_acq(mem, newval, oldval)            \
  (abort (), 0)

#define __arch_compare_and_exchange_bool_16_acq(mem, newval, oldval)           \
  (abort (), 0)

#define __arch_compare_and_exchange_val_32_acq(mem, newval, oldval)            \
  ({                                                                           \
      __typeof (*(mem)) __tmp;                                                 \
      __typeof (mem)  __memp = (mem);                                          \
      int test;                                                                \
      __asm __volatile (                                                       \
                "   addc    r0, r0, r0;"                                       \
                "1: lwx     %0, %3, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                "   cmp     %1, %0, %4;"                                       \
                "   bnei    %1, 2f;"                                           \
                "   swx     %5, %3, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                "2:"                                                           \
                    : "=&r" (__tmp),                                           \
                    "=&r" (test),                                              \
                    "=m" (*__memp)                                             \
                    : "r" (__memp),                                            \
                    "r" (oldval),                                              \
                    "r" (newval)                                               \
                    : "cc", "memory");                                         \
      __tmp;                                                                   \
  })

#define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval)            \
  (abort (), (__typeof (*mem)) 0)

#define atomic_compare_and_exchange_val_acq(mem, newval, oldval)               \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*mem) == 4)                                                    \
      __result = __arch_compare_and_exchange_val_32_acq (mem, newval, oldval); \
    else if (sizeof (*mem) == 8)                                               \
      __result = __arch_compare_and_exchange_val_64_acq (mem, newval, oldval); \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define atomic_compare_and_exchange_val_rel(mem, newval, oldval)               \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*mem) == 4)                                                    \
      __result = __arch_compare_and_exchange_val_32_acq (mem, newval, oldval); \
    else if (sizeof (*mem) == 8)                                               \
      __result = __arch_compare_and_exchange_val_64_acq (mem, newval, oldval); \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define __arch_atomic_exchange_32_acq(mem, value)                              \
  ({                                                                           \
      __typeof (*(mem)) __tmp;                                                 \
      __typeof (mem)  __memp = (mem);                                          \
      int test;                                                                \
      __asm __volatile (                                                       \
                "   addc    r0, r0, r0;"                                       \
                "1: lwx     %0, %4, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                "   swx     %3, %4, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                    : "=&r" (__tmp),                                           \
                    "=&r" (test),                                              \
                    "=m" (*__memp)                                             \
                    : "r" (value),                                             \
                    "r" (__memp)                                               \
                    : "cc", "memory");                                         \
      __tmp;                                                                   \
  })

#define __arch_atomic_exchange_64_acq(mem, newval)                             \
  (abort (), (__typeof (*mem)) 0)

#define atomic_exchange_acq(mem, value)                                        \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*mem) == 4)                                                    \
      __result = __arch_atomic_exchange_32_acq (mem, value);                   \
    else if (sizeof (*mem) == 8)                                               \
      __result = __arch_atomic_exchange_64_acq (mem, value);                   \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define atomic_exchange_rel(mem, value)                                        \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*mem) == 4)                                                    \
      __result = __arch_atomic_exchange_32_acq (mem, value);                   \
    else if (sizeof (*mem) == 8)                                               \
      __result = __arch_atomic_exchange_64_acq (mem, value);                   \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define __arch_atomic_exchange_and_add_32(mem, value)                          \
  ({                                                                           \
    __typeof (*(mem)) __tmp;                                                   \
      __typeof (mem)  __memp = (mem);                                          \
    int test;                                                                  \
    __asm __volatile (                                                         \
                "   addc    r0, r0, r0;"                                       \
                "1: lwx     %0, %4, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                "   add     %1, %3, %0;"                                       \
                "   swx     %1, %4, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                    : "=&r" (__tmp),                                           \
                    "=&r" (test),                                              \
                    "=m" (*__memp)                                             \
                    : "r" (value),                                             \
                    "r" (__memp)                                               \
                    : "cc", "memory");                                         \
    __tmp;                                                                     \
  })

#define __arch_atomic_exchange_and_add_64(mem, value)                          \
  (abort (), (__typeof (*mem)) 0)

#define atomic_exchange_and_add(mem, value)                                    \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*mem) == 4)                                                    \
      __result = __arch_atomic_exchange_and_add_32 (mem, value);               \
    else if (sizeof (*mem) == 8)                                               \
      __result = __arch_atomic_exchange_and_add_64 (mem, value);               \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define __arch_atomic_increment_val_32(mem)                                    \
  ({                                                                           \
    __typeof (*(mem)) __val;                                                   \
    int test;                                                                  \
    __asm __volatile (                                                         \
                "   addc    r0, r0, r0;"                                       \
                "1: lwx     %0, %3, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                "   addi    %0, %0, 1;"                                        \
                "   swx     %0, %3, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                    : "=&r" (__val),                                           \
                    "=&r" (test),                                              \
                    "=m" (*mem)                                                \
                    : "r" (mem),                                               \
                    "m" (*mem)                                                 \
                    : "cc", "memory");                                         \
    __val;                                                                     \
  })

#define __arch_atomic_increment_val_64(mem)                                    \
  (abort (), (__typeof (*mem)) 0)

#define atomic_increment_val(mem)                                              \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*(mem)) == 4)                                                  \
      __result = __arch_atomic_increment_val_32 (mem);                         \
    else if (sizeof (*(mem)) == 8)                                             \
      __result = __arch_atomic_increment_val_64 (mem);                         \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define atomic_increment(mem) ({ atomic_increment_val (mem); (void) 0; })

#define __arch_atomic_decrement_val_32(mem)                                    \
  ({                                                                           \
    __typeof (*(mem)) __val;                                                   \
    int test;                                                                  \
    __asm __volatile (                                                         \
                "   addc    r0, r0, r0;"                                       \
                "1: lwx     %0, %3, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                "   rsubi   %0, %0, 1;"                                        \
                "   swx     %0, %3, r0;"                                       \
                "   addic   %1, r0, 0;"                                        \
                "   bnei    %1, 1b;"                                           \
                    : "=&r" (__val),                                           \
                    "=&r" (test),                                              \
                    "=m" (*mem)                                                \
                    : "r" (mem),                                               \
                    "m" (*mem)                                                 \
                    : "cc", "memory");                                         \
    __val;                                                                     \
  })

#define __arch_atomic_decrement_val_64(mem)                                    \
  (abort (), (__typeof (*mem)) 0)

#define atomic_decrement_val(mem)                                              \
  ({                                                                           \
    __typeof (*(mem)) __result;                                                \
    if (sizeof (*(mem)) == 4)                                                  \
      __result = __arch_atomic_decrement_val_32 (mem);                         \
    else if (sizeof (*(mem)) == 8)                                             \
      __result = __arch_atomic_decrement_val_64 (mem);                         \
    else                                                                       \
       abort ();                                                               \
    __result;                                                                  \
  })

#define atomic_decrement(mem) ({ atomic_decrement_val (mem); (void) 0; })
