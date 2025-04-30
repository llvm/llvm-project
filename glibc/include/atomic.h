/* Internal macros for atomic operations for GNU C Library.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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

#ifndef _ATOMIC_H
#define _ATOMIC_H	1

/* This header defines three types of macros:

   - atomic arithmetic and logic operation on memory.  They all
     have the prefix "atomic_".

   - conditionally atomic operations of the same kinds.  These
     always behave identical but can be faster when atomicity
     is not really needed since only one thread has access to
     the memory location.  In that case the code is slower in
     the multi-thread case.  The interfaces have the prefix
     "catomic_".

   - support functions like barriers.  They also have the prefix
     "atomic_".

   Architectures must provide a few lowlevel macros (the compare
   and exchange definitions).  All others are optional.  They
   should only be provided if the architecture has specific
   support for the operation.

   As <atomic.h> macros are usually heavily nested and often use local
   variables to make sure side-effects are evaluated properly, use for
   macro local variables a per-macro unique prefix.  This file uses
   __atgN_ prefix where N is different in each macro.  */

#include <stdlib.h>

#include <atomic-machine.h>

/* Wrapper macros to call pre_NN_post (mem, ...) where NN is the
   bit width of *MEM.  The calling macro puts parens around MEM
   and following args.  */
#define __atomic_val_bysize(pre, post, mem, ...)			      \
  ({									      \
    __typeof ((__typeof (*(mem))) *(mem)) __atg1_result;		      \
    if (sizeof (*mem) == 1)						      \
      __atg1_result = pre##_8_##post (mem, __VA_ARGS__);		      \
    else if (sizeof (*mem) == 2)					      \
      __atg1_result = pre##_16_##post (mem, __VA_ARGS__);		      \
    else if (sizeof (*mem) == 4)					      \
      __atg1_result = pre##_32_##post (mem, __VA_ARGS__);		      \
    else if (sizeof (*mem) == 8)					      \
      __atg1_result = pre##_64_##post (mem, __VA_ARGS__);		      \
    else								      \
      abort ();								      \
    __atg1_result;							      \
  })
#define __atomic_bool_bysize(pre, post, mem, ...)			      \
  ({									      \
    int __atg2_result;							      \
    if (sizeof (*mem) == 1)						      \
      __atg2_result = pre##_8_##post (mem, __VA_ARGS__);		      \
    else if (sizeof (*mem) == 2)					      \
      __atg2_result = pre##_16_##post (mem, __VA_ARGS__);		      \
    else if (sizeof (*mem) == 4)					      \
      __atg2_result = pre##_32_##post (mem, __VA_ARGS__);		      \
    else if (sizeof (*mem) == 8)					      \
      __atg2_result = pre##_64_##post (mem, __VA_ARGS__);		      \
    else								      \
      abort ();								      \
    __atg2_result;							      \
  })


/* Atomically store NEWVAL in *MEM if *MEM is equal to OLDVAL.
   Return the old *MEM value.  */
#if !defined atomic_compare_and_exchange_val_acq \
    && defined __arch_compare_and_exchange_val_32_acq
# define atomic_compare_and_exchange_val_acq(mem, newval, oldval) \
  __atomic_val_bysize (__arch_compare_and_exchange_val,acq,		      \
		       mem, newval, oldval)
#endif


#ifndef catomic_compare_and_exchange_val_acq
# ifdef __arch_c_compare_and_exchange_val_32_acq
#  define catomic_compare_and_exchange_val_acq(mem, newval, oldval) \
  __atomic_val_bysize (__arch_c_compare_and_exchange_val,acq,		      \
		       mem, newval, oldval)
# else
#  define catomic_compare_and_exchange_val_acq(mem, newval, oldval) \
  atomic_compare_and_exchange_val_acq (mem, newval, oldval)
# endif
#endif


#ifndef catomic_compare_and_exchange_val_rel
# ifndef atomic_compare_and_exchange_val_rel
#  define catomic_compare_and_exchange_val_rel(mem, newval, oldval)	      \
  catomic_compare_and_exchange_val_acq (mem, newval, oldval)
# else
#  define catomic_compare_and_exchange_val_rel(mem, newval, oldval)	      \
  atomic_compare_and_exchange_val_rel (mem, newval, oldval)
# endif
#endif


#ifndef atomic_compare_and_exchange_val_rel
# define atomic_compare_and_exchange_val_rel(mem, newval, oldval)	      \
  atomic_compare_and_exchange_val_acq (mem, newval, oldval)
#endif


/* Atomically store NEWVAL in *MEM if *MEM is equal to OLDVAL.
   Return zero if *MEM was changed or non-zero if no exchange happened.  */
#ifndef atomic_compare_and_exchange_bool_acq
# ifdef __arch_compare_and_exchange_bool_32_acq
#  define atomic_compare_and_exchange_bool_acq(mem, newval, oldval) \
  __atomic_bool_bysize (__arch_compare_and_exchange_bool,acq,		      \
		        mem, newval, oldval)
# else
#  define atomic_compare_and_exchange_bool_acq(mem, newval, oldval) \
  ({ /* Cannot use __oldval here, because macros later in this file might     \
	call this macro with __oldval argument.	 */			      \
     __typeof (oldval) __atg3_old = (oldval);				      \
     atomic_compare_and_exchange_val_acq (mem, newval, __atg3_old)	      \
       != __atg3_old;							      \
  })
# endif
#endif


#ifndef catomic_compare_and_exchange_bool_acq
# ifdef __arch_c_compare_and_exchange_bool_32_acq
#  define catomic_compare_and_exchange_bool_acq(mem, newval, oldval) \
  __atomic_bool_bysize (__arch_c_compare_and_exchange_bool,acq,		      \
		        mem, newval, oldval)
# else
#  define catomic_compare_and_exchange_bool_acq(mem, newval, oldval) \
  ({ /* Cannot use __oldval here, because macros later in this file might     \
	call this macro with __oldval argument.	 */			      \
     __typeof (oldval) __atg4_old = (oldval);				      \
     catomic_compare_and_exchange_val_acq (mem, newval, __atg4_old)	      \
       != __atg4_old;							      \
  })
# endif
#endif


/* Store NEWVALUE in *MEM and return the old value.  */
#ifndef atomic_exchange_acq
# define atomic_exchange_acq(mem, newvalue) \
  ({ __typeof ((__typeof (*(mem))) *(mem)) __atg5_oldval;		      \
     __typeof (mem) __atg5_memp = (mem);				      \
     __typeof ((__typeof (*(mem))) *(mem)) __atg5_value = (newvalue);	      \
									      \
     do									      \
       __atg5_oldval = *__atg5_memp;					      \
     while (__builtin_expect						      \
	    (atomic_compare_and_exchange_bool_acq (__atg5_memp, __atg5_value, \
						   __atg5_oldval), 0));	      \
									      \
     __atg5_oldval; })
#endif

#ifndef atomic_exchange_rel
# define atomic_exchange_rel(mem, newvalue) atomic_exchange_acq (mem, newvalue)
#endif


/* Add VALUE to *MEM and return the old value of *MEM.  */
#ifndef atomic_exchange_and_add_acq
# ifdef atomic_exchange_and_add
#  define atomic_exchange_and_add_acq(mem, value) \
  atomic_exchange_and_add (mem, value)
# else
#  define atomic_exchange_and_add_acq(mem, value) \
  ({ __typeof (*(mem)) __atg6_oldval;					      \
     __typeof (mem) __atg6_memp = (mem);				      \
     __typeof (*(mem)) __atg6_value = (value);				      \
									      \
     do									      \
       __atg6_oldval = *__atg6_memp;					      \
     while (__builtin_expect						      \
	    (atomic_compare_and_exchange_bool_acq (__atg6_memp,		      \
						   __atg6_oldval	      \
						   + __atg6_value,	      \
						   __atg6_oldval), 0));	      \
									      \
     __atg6_oldval; })
# endif
#endif

#ifndef atomic_exchange_and_add_rel
# define atomic_exchange_and_add_rel(mem, value) \
  atomic_exchange_and_add_acq(mem, value)
#endif

#ifndef atomic_exchange_and_add
# define atomic_exchange_and_add(mem, value) \
  atomic_exchange_and_add_acq(mem, value)
#endif

#ifndef catomic_exchange_and_add
# define catomic_exchange_and_add(mem, value) \
  ({ __typeof (*(mem)) __atg7_oldv;					      \
     __typeof (mem) __atg7_memp = (mem);				      \
     __typeof (*(mem)) __atg7_value = (value);				      \
									      \
     do									      \
       __atg7_oldv = *__atg7_memp;					      \
     while (__builtin_expect						      \
	    (catomic_compare_and_exchange_bool_acq (__atg7_memp,	      \
						    __atg7_oldv		      \
						    + __atg7_value,	      \
						    __atg7_oldv), 0));	      \
									      \
     __atg7_oldv; })
#endif


#ifndef atomic_max
# define atomic_max(mem, value) \
  do {									      \
    __typeof (*(mem)) __atg8_oldval;					      \
    __typeof (mem) __atg8_memp = (mem);					      \
    __typeof (*(mem)) __atg8_value = (value);				      \
    do {								      \
      __atg8_oldval = *__atg8_memp;					      \
      if (__atg8_oldval >= __atg8_value)				      \
	break;								      \
    } while (__builtin_expect						      \
	     (atomic_compare_and_exchange_bool_acq (__atg8_memp, __atg8_value,\
						    __atg8_oldval), 0));      \
  } while (0)
#endif


#ifndef catomic_max
# define catomic_max(mem, value) \
  do {									      \
    __typeof (*(mem)) __atg9_oldv;					      \
    __typeof (mem) __atg9_memp = (mem);					      \
    __typeof (*(mem)) __atg9_value = (value);				      \
    do {								      \
      __atg9_oldv = *__atg9_memp;					      \
      if (__atg9_oldv >= __atg9_value)					      \
	break;								      \
    } while (__builtin_expect						      \
	     (catomic_compare_and_exchange_bool_acq (__atg9_memp,	      \
						     __atg9_value,	      \
						     __atg9_oldv), 0));	      \
  } while (0)
#endif


#ifndef atomic_min
# define atomic_min(mem, value) \
  do {									      \
    __typeof (*(mem)) __atg10_oldval;					      \
    __typeof (mem) __atg10_memp = (mem);				      \
    __typeof (*(mem)) __atg10_value = (value);				      \
    do {								      \
      __atg10_oldval = *__atg10_memp;					      \
      if (__atg10_oldval <= __atg10_value)				      \
	break;								      \
    } while (__builtin_expect						      \
	     (atomic_compare_and_exchange_bool_acq (__atg10_memp,	      \
						    __atg10_value,	      \
						    __atg10_oldval), 0));     \
  } while (0)
#endif


#ifndef atomic_add
# define atomic_add(mem, value) (void) atomic_exchange_and_add ((mem), (value))
#endif


#ifndef catomic_add
# define catomic_add(mem, value) \
  (void) catomic_exchange_and_add ((mem), (value))
#endif


#ifndef atomic_increment
# define atomic_increment(mem) atomic_add ((mem), 1)
#endif


#ifndef catomic_increment
# define catomic_increment(mem) catomic_add ((mem), 1)
#endif


#ifndef atomic_increment_val
# define atomic_increment_val(mem) (atomic_exchange_and_add ((mem), 1) + 1)
#endif


#ifndef catomic_increment_val
# define catomic_increment_val(mem) (catomic_exchange_and_add ((mem), 1) + 1)
#endif


/* Add one to *MEM and return true iff it's now zero.  */
#ifndef atomic_increment_and_test
# define atomic_increment_and_test(mem) \
  (atomic_exchange_and_add ((mem), 1) + 1 == 0)
#endif


#ifndef atomic_decrement
# define atomic_decrement(mem) atomic_add ((mem), -1)
#endif


#ifndef catomic_decrement
# define catomic_decrement(mem) catomic_add ((mem), -1)
#endif


#ifndef atomic_decrement_val
# define atomic_decrement_val(mem) (atomic_exchange_and_add ((mem), -1) - 1)
#endif


#ifndef catomic_decrement_val
# define catomic_decrement_val(mem) (catomic_exchange_and_add ((mem), -1) - 1)
#endif


/* Subtract 1 from *MEM and return true iff it's now zero.  */
#ifndef atomic_decrement_and_test
# define atomic_decrement_and_test(mem) \
  (atomic_exchange_and_add ((mem), -1) == 1)
#endif


/* Decrement *MEM if it is > 0, and return the old value.  */
#ifndef atomic_decrement_if_positive
# define atomic_decrement_if_positive(mem) \
  ({ __typeof (*(mem)) __atg11_oldval;					      \
     __typeof (mem) __atg11_memp = (mem);				      \
									      \
     do									      \
       {								      \
	 __atg11_oldval = *__atg11_memp;				      \
	 if (__glibc_unlikely (__atg11_oldval <= 0))			      \
	   break;							      \
       }								      \
     while (__builtin_expect						      \
	    (atomic_compare_and_exchange_bool_acq (__atg11_memp,	      \
						   __atg11_oldval - 1,	      \
						   __atg11_oldval), 0));      \
     __atg11_oldval; })
#endif


#ifndef atomic_add_negative
# define atomic_add_negative(mem, value)				      \
  ({ __typeof (value) __atg12_value = (value);				      \
     atomic_exchange_and_add (mem, __atg12_value) < -__atg12_value; })
#endif


#ifndef atomic_add_zero
# define atomic_add_zero(mem, value)					      \
  ({ __typeof (value) __atg13_value = (value);				      \
     atomic_exchange_and_add (mem, __atg13_value) == -__atg13_value; })
#endif


#ifndef atomic_bit_set
# define atomic_bit_set(mem, bit) \
  (void) atomic_bit_test_set(mem, bit)
#endif


#ifndef atomic_bit_test_set
# define atomic_bit_test_set(mem, bit) \
  ({ __typeof (*(mem)) __atg14_old;					      \
     __typeof (mem) __atg14_memp = (mem);				      \
     __typeof (*(mem)) __atg14_mask = ((__typeof (*(mem))) 1 << (bit));	      \
									      \
     do									      \
       __atg14_old = (*__atg14_memp);					      \
     while (__builtin_expect						      \
	    (atomic_compare_and_exchange_bool_acq (__atg14_memp,	      \
						   __atg14_old | __atg14_mask,\
						   __atg14_old), 0));	      \
									      \
     __atg14_old & __atg14_mask; })
#endif

/* Atomically *mem &= mask.  */
#ifndef atomic_and
# define atomic_and(mem, mask) \
  do {									      \
    __typeof (*(mem)) __atg15_old;					      \
    __typeof (mem) __atg15_memp = (mem);				      \
    __typeof (*(mem)) __atg15_mask = (mask);				      \
									      \
    do									      \
      __atg15_old = (*__atg15_memp);					      \
    while (__builtin_expect						      \
	   (atomic_compare_and_exchange_bool_acq (__atg15_memp,		      \
						  __atg15_old & __atg15_mask, \
						  __atg15_old), 0));	      \
  } while (0)
#endif

#ifndef catomic_and
# define catomic_and(mem, mask) \
  do {									      \
    __typeof (*(mem)) __atg20_old;					      \
    __typeof (mem) __atg20_memp = (mem);				      \
    __typeof (*(mem)) __atg20_mask = (mask);				      \
									      \
    do									      \
      __atg20_old = (*__atg20_memp);					      \
    while (__builtin_expect						      \
	   (catomic_compare_and_exchange_bool_acq (__atg20_memp,	      \
						   __atg20_old & __atg20_mask,\
						   __atg20_old), 0));	      \
  } while (0)
#endif

/* Atomically *mem &= mask and return the old value of *mem.  */
#ifndef atomic_and_val
# define atomic_and_val(mem, mask) \
  ({ __typeof (*(mem)) __atg16_old;					      \
     __typeof (mem) __atg16_memp = (mem);				      \
     __typeof (*(mem)) __atg16_mask = (mask);				      \
									      \
     do									      \
       __atg16_old = (*__atg16_memp);					      \
     while (__builtin_expect						      \
	    (atomic_compare_and_exchange_bool_acq (__atg16_memp,	      \
						   __atg16_old & __atg16_mask,\
						   __atg16_old), 0));	      \
									      \
     __atg16_old; })
#endif

/* Atomically *mem |= mask and return the old value of *mem.  */
#ifndef atomic_or
# define atomic_or(mem, mask) \
  do {									      \
    __typeof (*(mem)) __atg17_old;					      \
    __typeof (mem) __atg17_memp = (mem);				      \
    __typeof (*(mem)) __atg17_mask = (mask);				      \
									      \
    do									      \
      __atg17_old = (*__atg17_memp);					      \
    while (__builtin_expect						      \
	   (atomic_compare_and_exchange_bool_acq (__atg17_memp,		      \
						  __atg17_old | __atg17_mask, \
						  __atg17_old), 0));	      \
  } while (0)
#endif

#ifndef catomic_or
# define catomic_or(mem, mask) \
  do {									      \
    __typeof (*(mem)) __atg18_old;					      \
    __typeof (mem) __atg18_memp = (mem);				      \
    __typeof (*(mem)) __atg18_mask = (mask);				      \
									      \
    do									      \
      __atg18_old = (*__atg18_memp);					      \
    while (__builtin_expect						      \
	   (catomic_compare_and_exchange_bool_acq (__atg18_memp,	      \
						   __atg18_old | __atg18_mask,\
						   __atg18_old), 0));	      \
  } while (0)
#endif

/* Atomically *mem |= mask and return the old value of *mem.  */
#ifndef atomic_or_val
# define atomic_or_val(mem, mask) \
  ({ __typeof (*(mem)) __atg19_old;					      \
     __typeof (mem) __atg19_memp = (mem);				      \
     __typeof (*(mem)) __atg19_mask = (mask);				      \
									      \
     do									      \
       __atg19_old = (*__atg19_memp);					      \
     while (__builtin_expect						      \
	    (atomic_compare_and_exchange_bool_acq (__atg19_memp,	      \
						   __atg19_old | __atg19_mask,\
						   __atg19_old), 0));	      \
									      \
     __atg19_old; })
#endif

#ifndef atomic_full_barrier
# define atomic_full_barrier() __asm ("" ::: "memory")
#endif


#ifndef atomic_read_barrier
# define atomic_read_barrier() atomic_full_barrier ()
#endif


#ifndef atomic_write_barrier
# define atomic_write_barrier() atomic_full_barrier ()
#endif


#ifndef atomic_forced_read
# define atomic_forced_read(x) \
  ({ __typeof (x) __x; __asm ("" : "=r" (__x) : "0" (x)); __x; })
#endif

/* This is equal to 1 iff the architecture supports 64b atomic operations.  */
#ifndef __HAVE_64B_ATOMICS
#error Unable to determine if 64-bit atomics are present.
#endif

/* The following functions are a subset of the atomic operations provided by
   C11.  Usually, a function named atomic_OP_MO(args) is equivalent to C11's
   atomic_OP_explicit(args, memory_order_MO); exceptions noted below.  */

/* Each arch can request to use compiler built-ins for C11 atomics.  If it
   does, all atomics will be based on these.  */
#if USE_ATOMIC_COMPILER_BUILTINS

/* We require 32b atomic operations; some archs also support 64b atomic
   operations.  */
void __atomic_link_error (void);
# if __HAVE_64B_ATOMICS == 1
#  define __atomic_check_size(mem) \
   if ((sizeof (*mem) != 4) && (sizeof (*mem) != 8))			      \
     __atomic_link_error ();
# else
#  define __atomic_check_size(mem) \
   if (sizeof (*mem) != 4)						      \
     __atomic_link_error ();
# endif
/* We additionally provide 8b and 16b atomic loads and stores; we do not yet
   need other atomic operations of such sizes, and restricting the support to
   loads and stores makes this easier for archs that do not have native
   support for atomic operations to less-than-word-sized data.  */
# if __HAVE_64B_ATOMICS == 1
#  define __atomic_check_size_ls(mem) \
   if ((sizeof (*mem) != 1) && (sizeof (*mem) != 2) && (sizeof (*mem) != 4)   \
       && (sizeof (*mem) != 8))						      \
     __atomic_link_error ();
# else
#  define __atomic_check_size_ls(mem) \
   if ((sizeof (*mem) != 1) && (sizeof (*mem) != 2) && sizeof (*mem) != 4)    \
     __atomic_link_error ();
# endif

# define atomic_thread_fence_acquire() \
  __atomic_thread_fence (__ATOMIC_ACQUIRE)
# define atomic_thread_fence_release() \
  __atomic_thread_fence (__ATOMIC_RELEASE)
# define atomic_thread_fence_seq_cst() \
  __atomic_thread_fence (__ATOMIC_SEQ_CST)

# define atomic_load_relaxed(mem) \
  ({ __atomic_check_size_ls((mem));					      \
     __atomic_load_n ((mem), __ATOMIC_RELAXED); })
# define atomic_load_acquire(mem) \
  ({ __atomic_check_size_ls((mem));					      \
     __atomic_load_n ((mem), __ATOMIC_ACQUIRE); })

# define atomic_store_relaxed(mem, val) \
  do {									      \
    __atomic_check_size_ls((mem));					      \
    __atomic_store_n ((mem), (val), __ATOMIC_RELAXED);			      \
  } while (0)
# define atomic_store_release(mem, val) \
  do {									      \
    __atomic_check_size_ls((mem));					      \
    __atomic_store_n ((mem), (val), __ATOMIC_RELEASE);			      \
  } while (0)

/* On failure, this CAS has memory_order_relaxed semantics.  */
# define atomic_compare_exchange_weak_relaxed(mem, expected, desired) \
  ({ __atomic_check_size((mem));					      \
  __atomic_compare_exchange_n ((mem), (expected), (desired), 1,		      \
    __ATOMIC_RELAXED, __ATOMIC_RELAXED); })
# define atomic_compare_exchange_weak_acquire(mem, expected, desired) \
  ({ __atomic_check_size((mem));					      \
  __atomic_compare_exchange_n ((mem), (expected), (desired), 1,		      \
    __ATOMIC_ACQUIRE, __ATOMIC_RELAXED); })
# define atomic_compare_exchange_weak_release(mem, expected, desired) \
  ({ __atomic_check_size((mem));					      \
  __atomic_compare_exchange_n ((mem), (expected), (desired), 1,		      \
    __ATOMIC_RELEASE, __ATOMIC_RELAXED); })

# define atomic_exchange_relaxed(mem, desired) \
  ({ __atomic_check_size((mem));					      \
  __atomic_exchange_n ((mem), (desired), __ATOMIC_RELAXED); })
# define atomic_exchange_acquire(mem, desired) \
  ({ __atomic_check_size((mem));					      \
  __atomic_exchange_n ((mem), (desired), __ATOMIC_ACQUIRE); })
# define atomic_exchange_release(mem, desired) \
  ({ __atomic_check_size((mem));					      \
  __atomic_exchange_n ((mem), (desired), __ATOMIC_RELEASE); })

# define atomic_fetch_add_relaxed(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_add ((mem), (operand), __ATOMIC_RELAXED); })
# define atomic_fetch_add_acquire(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_add ((mem), (operand), __ATOMIC_ACQUIRE); })
# define atomic_fetch_add_release(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_add ((mem), (operand), __ATOMIC_RELEASE); })
# define atomic_fetch_add_acq_rel(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_add ((mem), (operand), __ATOMIC_ACQ_REL); })

# define atomic_fetch_and_relaxed(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_and ((mem), (operand), __ATOMIC_RELAXED); })
# define atomic_fetch_and_acquire(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_and ((mem), (operand), __ATOMIC_ACQUIRE); })
# define atomic_fetch_and_release(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_and ((mem), (operand), __ATOMIC_RELEASE); })

# define atomic_fetch_or_relaxed(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_or ((mem), (operand), __ATOMIC_RELAXED); })
# define atomic_fetch_or_acquire(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_or ((mem), (operand), __ATOMIC_ACQUIRE); })
# define atomic_fetch_or_release(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_or ((mem), (operand), __ATOMIC_RELEASE); })

# define atomic_fetch_xor_release(mem, operand) \
  ({ __atomic_check_size((mem));					      \
  __atomic_fetch_xor ((mem), (operand), __ATOMIC_RELEASE); })

#else /* !USE_ATOMIC_COMPILER_BUILTINS  */

/* By default, we assume that read, write, and full barriers are equivalent
   to acquire, release, and seq_cst barriers.  Archs for which this does not
   hold have to provide custom definitions of the fences.  */
# ifndef atomic_thread_fence_acquire
#  define atomic_thread_fence_acquire() atomic_read_barrier ()
# endif
# ifndef atomic_thread_fence_release
#  define atomic_thread_fence_release() atomic_write_barrier ()
# endif
# ifndef atomic_thread_fence_seq_cst
#  define atomic_thread_fence_seq_cst() atomic_full_barrier ()
# endif

# ifndef atomic_load_relaxed
#  define atomic_load_relaxed(mem) \
   ({ __typeof ((__typeof (*(mem))) *(mem)) __atg100_val;		      \
   __asm ("" : "=r" (__atg100_val) : "0" (*(mem)));			      \
   __atg100_val; })
# endif
# ifndef atomic_load_acquire
#  define atomic_load_acquire(mem) \
   ({ __typeof (*(mem)) __atg101_val = atomic_load_relaxed (mem);	      \
   atomic_thread_fence_acquire ();					      \
   __atg101_val; })
# endif

# ifndef atomic_store_relaxed
/* XXX Use inline asm here?  */
#  define atomic_store_relaxed(mem, val) do { *(mem) = (val); } while (0)
# endif
# ifndef atomic_store_release
#  define atomic_store_release(mem, val) \
   do {									      \
     atomic_thread_fence_release ();					      \
     atomic_store_relaxed ((mem), (val));				      \
   } while (0)
# endif

/* On failure, this CAS has memory_order_relaxed semantics.  */
/* XXX This potentially has one branch more than necessary, but archs
   currently do not define a CAS that returns both the previous value and
   the success flag.  */
# ifndef atomic_compare_exchange_weak_acquire
#  define atomic_compare_exchange_weak_acquire(mem, expected, desired) \
   ({ typeof (*(expected)) __atg102_expected = *(expected);		      \
   *(expected) =							      \
     atomic_compare_and_exchange_val_acq ((mem), (desired), *(expected));     \
   *(expected) == __atg102_expected; })
# endif
# ifndef atomic_compare_exchange_weak_relaxed
/* XXX Fall back to CAS with acquire MO because archs do not define a weaker
   CAS.  */
#  define atomic_compare_exchange_weak_relaxed(mem, expected, desired) \
   atomic_compare_exchange_weak_acquire ((mem), (expected), (desired))
# endif
# ifndef atomic_compare_exchange_weak_release
#  define atomic_compare_exchange_weak_release(mem, expected, desired) \
   ({ typeof (*(expected)) __atg103_expected = *(expected);		      \
   *(expected) =							      \
     atomic_compare_and_exchange_val_rel ((mem), (desired), *(expected));     \
   *(expected) == __atg103_expected; })
# endif

/* XXX Fall back to acquire MO because archs do not define a weaker
   atomic_exchange.  */
# ifndef atomic_exchange_relaxed
#  define atomic_exchange_relaxed(mem, val) \
   atomic_exchange_acq ((mem), (val))
# endif
# ifndef atomic_exchange_acquire
#  define atomic_exchange_acquire(mem, val) \
   atomic_exchange_acq ((mem), (val))
# endif
# ifndef atomic_exchange_release
#  define atomic_exchange_release(mem, val) \
   atomic_exchange_rel ((mem), (val))
# endif

# ifndef atomic_fetch_add_acquire
#  define atomic_fetch_add_acquire(mem, operand) \
   atomic_exchange_and_add_acq ((mem), (operand))
# endif
# ifndef atomic_fetch_add_relaxed
/* XXX Fall back to acquire MO because the MO semantics of
   atomic_exchange_and_add are not documented; the generic version falls back
   to atomic_exchange_and_add_acq if atomic_exchange_and_add is not defined,
   and vice versa.  */
#  define atomic_fetch_add_relaxed(mem, operand) \
   atomic_fetch_add_acquire ((mem), (operand))
# endif
# ifndef atomic_fetch_add_release
#  define atomic_fetch_add_release(mem, operand) \
   atomic_exchange_and_add_rel ((mem), (operand))
# endif
# ifndef atomic_fetch_add_acq_rel
#  define atomic_fetch_add_acq_rel(mem, operand) \
   ({ atomic_thread_fence_release ();					      \
   atomic_exchange_and_add_acq ((mem), (operand)); })
# endif

/* XXX Fall back to acquire MO because archs do not define a weaker
   atomic_and_val.  */
# ifndef atomic_fetch_and_relaxed
#  define atomic_fetch_and_relaxed(mem, operand) \
   atomic_fetch_and_acquire ((mem), (operand))
# endif
/* XXX The default for atomic_and_val has acquire semantics, but this is not
   documented.  */
# ifndef atomic_fetch_and_acquire
#  define atomic_fetch_and_acquire(mem, operand) \
   atomic_and_val ((mem), (operand))
# endif
# ifndef atomic_fetch_and_release
/* XXX This unnecessarily has acquire MO.  */
#  define atomic_fetch_and_release(mem, operand) \
   ({ atomic_thread_fence_release ();					      \
   atomic_and_val ((mem), (operand)); })
# endif

/* XXX The default for atomic_or_val has acquire semantics, but this is not
   documented.  */
# ifndef atomic_fetch_or_acquire
#  define atomic_fetch_or_acquire(mem, operand) \
   atomic_or_val ((mem), (operand))
# endif
/* XXX Fall back to acquire MO because archs do not define a weaker
   atomic_or_val.  */
# ifndef atomic_fetch_or_relaxed
#  define atomic_fetch_or_relaxed(mem, operand) \
   atomic_fetch_or_acquire ((mem), (operand))
# endif
/* XXX Contains an unnecessary acquire MO because archs do not define a weaker
   atomic_or_val.  */
# ifndef atomic_fetch_or_release
#  define atomic_fetch_or_release(mem, operand) \
   ({ atomic_thread_fence_release ();					      \
   atomic_fetch_or_acquire ((mem), (operand)); })
# endif

# ifndef atomic_fetch_xor_release
/* Failing the atomic_compare_exchange_weak_release reloads the value in
   __atg104_expected, so we need only do the XOR again and retry.  */
# define atomic_fetch_xor_release(mem, operand) \
  ({ __typeof (mem) __atg104_memp = (mem);				      \
     __typeof (*(mem)) __atg104_expected = (*__atg104_memp);		      \
     __typeof (*(mem)) __atg104_desired;				      \
     __typeof (*(mem)) __atg104_op = (operand);				      \
									      \
     do									      \
       __atg104_desired = __atg104_expected ^ __atg104_op;		      \
     while (__glibc_unlikely						      \
	    (atomic_compare_exchange_weak_release (			      \
	       __atg104_memp, &__atg104_expected, __atg104_desired)	      \
	     == 0));							      \
     __atg104_expected; })
#endif

#endif /* !USE_ATOMIC_COMPILER_BUILTINS  */

/* This operation does not affect synchronization semantics but can be used
   in the body of a spin loop to potentially improve its efficiency.  */
#ifndef atomic_spin_nop
# define atomic_spin_nop() do { /* nothing */ } while (0)
#endif

/* ATOMIC_EXCHANGE_USES_CAS is non-zero if atomic_exchange operations
   are implemented based on a CAS loop; otherwise, this is zero and we assume
   that the atomic_exchange operations could provide better performance
   than a CAS loop.  */
#ifndef ATOMIC_EXCHANGE_USES_CAS
# error ATOMIC_EXCHANGE_USES_CAS has to be defined.
#endif

#endif	/* atomic.h */
