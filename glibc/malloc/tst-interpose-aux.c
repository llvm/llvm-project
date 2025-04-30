/* Minimal malloc implementation for interposition tests.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include "tst-interpose-aux.h"

#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <time.h>

#if INTERPOSE_THREADS
#include <pthread.h>
#endif

/* Print the error message and terminate the process with status 1.  */
__attribute__ ((noreturn))
__attribute__ ((format (printf, 1, 2)))
static void *
fail (const char *format, ...)
{
  /* This assumes that vsnprintf will not call malloc.  It does not do
     so for the format strings we use.  */
  char message[4096];
  va_list ap;
  va_start (ap, format);
  vsnprintf (message, sizeof (message), format, ap);
  va_end (ap);

  enum { count = 3 };
  struct iovec iov[count];

  iov[0].iov_base = (char *) "error: ";
  iov[1].iov_base = (char *) message;
  iov[2].iov_base = (char *) "\n";

  for (int i = 0; i < count; ++i)
    iov[i].iov_len = strlen (iov[i].iov_base);

  int unused __attribute__ ((unused));
  unused = writev (STDOUT_FILENO, iov, count);
  _exit (1);
}

#if INTERPOSE_THREADS
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

static void
lock (void)
{
#if INTERPOSE_THREADS
  int ret = pthread_mutex_lock (&mutex);
  if (ret != 0)
    {
      errno = ret;
      fail ("pthread_mutex_lock: %m");
    }
#endif
}

static void
unlock (void)
{
#if INTERPOSE_THREADS
  int ret = pthread_mutex_unlock (&mutex);
  if (ret != 0)
    {
      errno = ret;
      fail ("pthread_mutex_unlock: %m");
    }
#endif
}

struct __attribute__ ((aligned (__alignof__ (max_align_t)))) allocation_header
{
  size_t allocation_index;
  size_t allocation_size;
  struct timespec ts;
};

/* Array of known allocations, to track invalid frees.  */
enum { max_allocations = 65536 };
static struct allocation_header *allocations[max_allocations];
static size_t allocation_index;
static size_t deallocation_count;

/* Sanity check for successful malloc interposition.  */
__attribute__ ((destructor))
static void
check_for_allocations (void)
{
  if (allocation_index == 0)
    {
      /* Make sure that malloc is called at least once from libc.  */
      void *volatile ptr = strdup ("ptr");
      /* Compiler barrier.  The strdup function calls malloc, which
         updates allocation_index, but strdup is marked __THROW, so
         the compiler could optimize away the reload.  */
      __asm__ volatile ("" ::: "memory");
      free (ptr);
      /* If the allocation count is still zero, it means we did not
         interpose malloc successfully.  */
      if (allocation_index == 0)
        fail ("malloc does not seem to have been interposed");
    }
}

static struct allocation_header *get_header (const char *op, void *ptr)
{
  struct allocation_header *header = ((struct allocation_header *) ptr) - 1;
  if (header->allocation_index >= allocation_index)
    fail ("%s: %p: invalid allocation index: %zu (not less than %zu)",
          op, ptr, header->allocation_index, allocation_index);
  if (allocations[header->allocation_index] != header)
    fail ("%s: %p: allocation pointer does not point to header, but %p",
          op, ptr, allocations[header->allocation_index]);
  return header;
}

/* Internal helper functions.  Those must be called while the lock is
   acquired.  */

static void *
malloc_internal (size_t size)
{
  if (allocation_index == max_allocations)
    {
      errno = ENOMEM;
      return NULL;
    }
  size_t allocation_size = size + sizeof (struct allocation_header);
  if (allocation_size < size)
    {
      errno = ENOMEM;
      return NULL;
    }

  size_t index = allocation_index++;
  void *result = mmap (NULL, allocation_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (result == MAP_FAILED)
    return NULL;
  allocations[index] = result;
  *allocations[index] = (struct allocation_header)
    {
      .allocation_index = index,
      .allocation_size = allocation_size
    };
  /* BZ#24967: Check if calling a symbol which may use the vDSO does not fail.
     The CLOCK_REALTIME should be supported on all systems.  */
  clock_gettime (CLOCK_REALTIME, &allocations[index]->ts);
  return allocations[index] + 1;
}

static void
free_internal (const char *op, struct allocation_header *header)
{
  size_t index = header->allocation_index;
  int result = mprotect (header, header->allocation_size, PROT_NONE);
  if (result != 0)
    fail ("%s: mprotect (%p, %zu): %m", op, header, header->allocation_size);
  /* Catch double-free issues.  */
  allocations[index] = NULL;
  ++deallocation_count;
}

static void *
realloc_internal (void *ptr, size_t new_size)
{
  struct allocation_header *header = get_header ("realloc", ptr);
  size_t old_size = header->allocation_size - sizeof (struct allocation_header);
  if (old_size >= new_size)
    return ptr;

  void *newptr = malloc_internal (new_size);
  if (newptr == NULL)
    return NULL;
  memcpy (newptr, ptr, old_size);
  free_internal ("realloc", header);
  return newptr;
}

/* Public interfaces.  These functions must perform locking.  */

size_t
malloc_allocation_count (void)
{
  lock ();
  size_t count = allocation_index;
  unlock ();
  return count;
}

size_t
malloc_deallocation_count (void)
{
  lock ();
  size_t count = deallocation_count;
  unlock ();
  return count;
}
void *
malloc (size_t size)
{
  lock ();
  void *result = malloc_internal (size);
  unlock ();
  return result;
}

void
free (void *ptr)
{
  if (ptr == NULL)
    return;
  lock ();
  struct allocation_header *header = get_header ("free", ptr);
  free_internal ("free", header);
  unlock ();
}

void *
calloc (size_t a, size_t b)
{
  if (b > 0 && a > SIZE_MAX / b)
    {
      errno = ENOMEM;
      return NULL;
    }
  lock ();
  /* malloc_internal uses mmap, so the memory is zeroed.  */
  void *result = malloc_internal (a * b);
  unlock ();
  return result;
}

void *
realloc (void *ptr, size_t n)
{
  if (n ==0)
    {
      free (ptr);
      return NULL;
    }
  else if (ptr == NULL)
    return malloc (n);
  else
    {
      lock ();
      void *result = realloc_internal (ptr, n);
      unlock ();
      return result;
    }
}
