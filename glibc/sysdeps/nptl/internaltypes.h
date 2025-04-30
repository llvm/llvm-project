/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#ifndef _INTERNALTYPES_H
#define _INTERNALTYPES_H	1

#include <stdint.h>
#include <atomic.h>
#include <endian.h>


struct pthread_attr
{
  /* Scheduler parameters and priority.  */
  struct sched_param schedparam;
  int schedpolicy;
  /* Various flags like detachstate, scope, etc.  */
  int flags;
  /* Size of guard area.  */
  size_t guardsize;
  /* Stack handling.  */
  void *stackaddr;
  size_t stacksize;

  /* Allocated via a call to __pthread_attr_extension once needed.  */
  struct pthread_attr_extension *extension;
  void *unused;
};

#define ATTR_FLAG_DETACHSTATE		0x0001
#define ATTR_FLAG_NOTINHERITSCHED	0x0002
#define ATTR_FLAG_SCOPEPROCESS		0x0004
#define ATTR_FLAG_STACKADDR		0x0008
#define ATTR_FLAG_OLDATTR		0x0010
#define ATTR_FLAG_SCHED_SET		0x0020
#define ATTR_FLAG_POLICY_SET		0x0040

/* Used to allocate a pthread_attr_t object which is also accessed
   internally.  */
union pthread_attr_transparent
{
  pthread_attr_t external;
  struct pthread_attr internal;
};

/* Extension space for pthread attributes.  Referenced by the
   extension member of struct pthread_attr.  */
struct pthread_attr_extension
{
  /* Affinity map.  */
  cpu_set_t *cpuset;
  size_t cpusetsize;

  sigset_t sigmask;
  bool sigmask_set;
};

/* Mutex attribute data structure.  */
struct pthread_mutexattr
{
  /* Identifier for the kind of mutex.

     Bit 31 is set if the mutex is to be shared between processes.

     Bit 0 to 30 contain one of the PTHREAD_MUTEX_ values to identify
     the type of the mutex.  */
  int mutexkind;
};


/* Conditional variable attribute data structure.  */
struct pthread_condattr
{
  /* Combination of values:

     Bit 0                : flag whether conditional variable will be
                            sharable between processes.
     Bit 1-COND_CLOCK_BITS: Clock ID.  COND_CLOCK_BITS is the number of bits
                            needed to represent the ID of the clock.  */
  int value;
};
#define COND_CLOCK_BITS	1


/* Read-write lock variable attribute data structure.  */
struct pthread_rwlockattr
{
  int lockkind;
  int pshared;
};


/* Barrier data structure.  See pthread_barrier_wait for a description
   of how these fields are used.  */
struct pthread_barrier
{
  unsigned int in;
  unsigned int current_round;
  unsigned int count;
  int shared;
  unsigned int out;
};
/* See pthread_barrier_wait for a description.  */
#define BARRIER_IN_THRESHOLD (UINT_MAX/2)


/* Barrier variable attribute data structure.  */
struct pthread_barrierattr
{
  int pshared;
};


/* Thread-local data handling.  */
struct pthread_key_struct
{
  /* Sequence numbers.  Even numbers indicated vacant entries.  Note
     that zero is even.  We use uintptr_t to not require padding on
     32- and 64-bit machines.  On 64-bit machines it helps to avoid
     wrapping, too.  */
  uintptr_t seq;

  /* Destructor for the data.  */
  void (*destr) (void *);
};

/* Check whether an entry is unused.  */
#define KEY_UNUSED(p) (((p) & 1) == 0)
/* Check whether a key is usable.  We cannot reuse an allocated key if
   the sequence counter would overflow after the next destroy call.
   This would mean that we potentially free memory for a key with the
   same sequence.  This is *very* unlikely to happen, A program would
   have to create and destroy a key 2^31 times (on 32-bit platforms,
   on 64-bit platforms that would be 2^63).  If it should happen we
   simply don't use this specific key anymore.  */
#define KEY_USABLE(p) (((uintptr_t) (p)) < ((uintptr_t) ((p) + 2)))


/* Handling of read-write lock data.  */
// XXX For now there is only one flag.  Maybe more in future.
#define RWLOCK_RECURSIVE(rwlock) ((rwlock)->__data.__flags != 0)


/* Semaphore variable structure.  */
struct new_sem
{
#if __HAVE_64B_ATOMICS
  /* The data field holds both value (in the least-significant 32 bits) and
     nwaiters.  */
# if __BYTE_ORDER == __LITTLE_ENDIAN
#  define SEM_VALUE_OFFSET 0
# elif __BYTE_ORDER == __BIG_ENDIAN
#  define SEM_VALUE_OFFSET 1
# else
# error Unsupported byte order.
# endif
# define SEM_NWAITERS_SHIFT 32
# define SEM_VALUE_MASK (~(unsigned int)0)
  uint64_t data;
  int private;
  int pad;
#else
# define SEM_VALUE_SHIFT 1
# define SEM_NWAITERS_MASK ((unsigned int)1)
  unsigned int value;
  int private;
  int pad;
  unsigned int nwaiters;
#endif
};

struct old_sem
{
  unsigned int value;
};


/* Compatibility type for old conditional variable interfaces.  */
typedef struct
{
  pthread_cond_t *cond;
} pthread_cond_2_0_t;

#endif	/* internaltypes.h */
