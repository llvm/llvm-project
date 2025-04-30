/* mcheck debugging hooks for malloc.
   Copyright (C) 1990-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written May 1989 by Mike Haertel.

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

#include <malloc-internal.h>
#include <mcheck.h>
#include <libintl.h>
#include <stdint.h>
#include <stdio.h>

/* Arbitrary magical numbers.  */
#define MAGICWORD       0xfedabeeb
#define MAGICFREE       0xd8675309
#define MAGICBYTE       ((char) 0xd7)
#define MALLOCFLOOD     ((char) 0x93)
#define FREEFLOOD       ((char) 0x95)

/* Function to call when something awful happens.  */
static void (*abortfunc) (enum mcheck_status);

struct hdr
{
  size_t size;                  /* Exact size requested by user.  */
  unsigned long int magic;      /* Magic number to check header integrity.  */
  struct hdr *prev;
  struct hdr *next;
  void *block;                  /* Real block allocated, for memalign.  */
  unsigned long int magic2;     /* Extra, keeps us doubleword aligned.  */
} __attribute__ ((aligned (MALLOC_ALIGNMENT)));

/* This is the beginning of the list of all memory blocks allocated.
   It is only constructed if the pedantic testing is requested.  */
static struct hdr *root;

/* Nonzero if pedentic checking of all blocks is requested.  */
static bool pedantic;

#if defined _LIBC || defined STDC_HEADERS || defined USG
# include <string.h>
# define flood memset
#else
static void flood (void *, int, size_t);
static void
flood (void *ptr, int val, size_t size)
{
  char *cp = ptr;
  while (size--)
    *cp++ = val;
}
#endif

static enum mcheck_status
checkhdr (const struct hdr *hdr)
{
  enum mcheck_status status;
  bool mcheck_used = __is_malloc_debug_enabled (MALLOC_MCHECK_HOOK);

  if (!mcheck_used)
    /* Maybe the mcheck used is disabled?  This happens when we find
       an error and report it.  */
    return MCHECK_OK;

  switch (hdr->magic ^ ((uintptr_t) hdr->prev + (uintptr_t) hdr->next))
    {
    default:
      status = MCHECK_HEAD;
      break;
    case MAGICFREE:
      status = MCHECK_FREE;
      break;
    case MAGICWORD:
      if (((char *) &hdr[1])[hdr->size] != MAGICBYTE)
	status = MCHECK_TAIL;
      else if ((hdr->magic2 ^ (uintptr_t) hdr->block) != MAGICWORD)
	status = MCHECK_HEAD;
      else
	status = MCHECK_OK;
      break;
    }
  if (status != MCHECK_OK)
    {
      mcheck_used = 0;
      (*abortfunc) (status);
      mcheck_used = 1;
    }
  return status;
}

static enum mcheck_status
__mcheck_checkptr (const void *ptr)
{
  if (!__is_malloc_debug_enabled (MALLOC_MCHECK_HOOK))
      return MCHECK_DISABLED;

  if (ptr != NULL)
    return checkhdr (((struct hdr *) ptr) - 1);

  /* Walk through all the active blocks and test whether they were tampered
     with.  */
  struct hdr *runp = root;

  /* Temporarily turn off the checks.  */
  pedantic = false;

  while (runp != NULL)
    {
      (void) checkhdr (runp);

      runp = runp->next;
    }

  /* Turn checks on again.  */
  pedantic = true;

  return MCHECK_OK;
}

static void
unlink_blk (struct hdr *ptr)
{
  if (ptr->next != NULL)
    {
      ptr->next->prev = ptr->prev;
      ptr->next->magic = MAGICWORD ^ ((uintptr_t) ptr->next->prev
                                      + (uintptr_t) ptr->next->next);
    }
  if (ptr->prev != NULL)
    {
      ptr->prev->next = ptr->next;
      ptr->prev->magic = MAGICWORD ^ ((uintptr_t) ptr->prev->prev
                                      + (uintptr_t) ptr->prev->next);
    }
  else
    root = ptr->next;
}

static void
link_blk (struct hdr *hdr)
{
  hdr->prev = NULL;
  hdr->next = root;
  root = hdr;
  hdr->magic = MAGICWORD ^ (uintptr_t) hdr->next;

  /* And the next block.  */
  if (hdr->next != NULL)
    {
      hdr->next->prev = hdr;
      hdr->next->magic = MAGICWORD ^ ((uintptr_t) hdr
                                      + (uintptr_t) hdr->next->next);
    }
}

static void *
free_mcheck (void *ptr)
{
  if (pedantic)
    __mcheck_checkptr (NULL);
  if (ptr)
    {
      struct hdr *hdr = ((struct hdr *) ptr) - 1;
      checkhdr (hdr);
      hdr->magic = MAGICFREE;
      hdr->magic2 = MAGICFREE;
      unlink_blk (hdr);
      hdr->prev = hdr->next = NULL;
      flood (ptr, FREEFLOOD, hdr->size);
      ptr = hdr->block;
    }
  return ptr;
}

static bool
malloc_mcheck_before (size_t *sizep, void **victimp)
{
  size_t size = *sizep;

  if (pedantic)
    __mcheck_checkptr (NULL);

  if (size > ~((size_t) 0) - (sizeof (struct hdr) + 1))
    {
      __set_errno (ENOMEM);
      *victimp = NULL;
      return true;
    }

  *sizep = sizeof (struct hdr) + size + 1;
  return false;
}

static void *
malloc_mcheck_after (void *mem, size_t size)
{
  struct hdr *hdr = mem;

  if (hdr == NULL)
    return NULL;

  hdr->size = size;
  link_blk (hdr);
  hdr->block = hdr;
  hdr->magic2 = (uintptr_t) hdr ^ MAGICWORD;
  ((char *) &hdr[1])[size] = MAGICBYTE;
  flood ((void *) (hdr + 1), MALLOCFLOOD, size);
  return (void *) (hdr + 1);
}

static bool
memalign_mcheck_before (size_t alignment, size_t *sizep, void **victimp)
{
  struct hdr *hdr;
  size_t slop, size = *sizep;

  /* Punt to malloc to avoid double headers.  */
  if (alignment <= MALLOC_ALIGNMENT)
    {
      *victimp = __debug_malloc (size);
      return true;
    }

  if (pedantic)
    __mcheck_checkptr (NULL);

  slop = (sizeof *hdr + alignment - 1) & - alignment;

  if (size > ~((size_t) 0) - (slop + 1))
    {
      __set_errno (ENOMEM);
      *victimp = NULL;
      return true;
    }

  *sizep = slop + size + 1;
  return false;
}

static void *
memalign_mcheck_after (void *block, size_t alignment, size_t size)
{
  if (block == NULL)
    return NULL;

  /* This was served by __debug_malloc, so return as is.  */
  if (alignment <= MALLOC_ALIGNMENT)
    return block;

  size_t slop = (sizeof (struct hdr) + alignment - 1) & - alignment;
  struct hdr *hdr = ((struct hdr *) (block + slop)) - 1;

  hdr->size = size;
  link_blk (hdr);
  hdr->block = (void *) block;
  hdr->magic2 = (uintptr_t) block ^ MAGICWORD;
  ((char *) &hdr[1])[size] = MAGICBYTE;
  flood ((void *) (hdr + 1), MALLOCFLOOD, size);
  return (void *) (hdr + 1);
}

static bool
realloc_mcheck_before (void **ptrp, size_t *sizep, size_t *oldsize,
		       void **victimp)
{
  size_t size = *sizep;
  void *ptr = *ptrp;

  if (ptr == NULL)
    {
      *victimp = __debug_malloc (size);
      *oldsize = 0;
      return true;
    }

  if (size == 0)
    {
      __debug_free (ptr);
      *victimp = NULL;
      return true;
    }

  if (size > ~((size_t) 0) - (sizeof (struct hdr) + 1))
    {
      __set_errno (ENOMEM);
      *victimp = NULL;
      *oldsize = 0;
      return true;
    }

  if (pedantic)
    __mcheck_checkptr (NULL);

  struct hdr *hdr;
  size_t osize;

  /* Update the oldptr for glibc realloc.  */
  *ptrp = hdr = ((struct hdr *) ptr) - 1;

  osize = hdr->size;

  checkhdr (hdr);
  unlink_blk (hdr);
  if (size < osize)
    flood ((char *) ptr + size, FREEFLOOD, osize - size);

  *oldsize = osize;
  *sizep = sizeof (struct hdr) + size + 1;
  return false;
}

static void *
realloc_mcheck_after (void *ptr, void *oldptr, size_t size, size_t osize)
{
  struct hdr *hdr = ptr;

  if (hdr == NULL)
    return NULL;

  /* Malloc already added the header so don't tamper with it.  */
  if (oldptr == NULL)
    return ptr;

  hdr->size = size;
  link_blk (hdr);
  hdr->block = hdr;
  hdr->magic2 = (uintptr_t) hdr ^ MAGICWORD;
  ((char *) &hdr[1])[size] = MAGICBYTE;
  if (size > osize)
    flood ((char *) (hdr + 1) + osize, MALLOCFLOOD, size - osize);
  return (void *) (hdr + 1);
}

__attribute__ ((noreturn))
static void
mabort (enum mcheck_status status)
{
  const char *msg;
  switch (status)
    {
    case MCHECK_OK:
      msg = _ ("memory is consistent, library is buggy\n");
      break;
    case MCHECK_HEAD:
      msg = _ ("memory clobbered before allocated block\n");
      break;
    case MCHECK_TAIL:
      msg = _ ("memory clobbered past end of allocated block\n");
      break;
    case MCHECK_FREE:
      msg = _ ("block freed twice\n");
      break;
    default:
      msg = _ ("bogus mcheck_status, library is buggy\n");
      break;
    }
#ifdef _LIBC
  __libc_fatal (msg);
#else
  fprintf (stderr, "mcheck: %s", msg);
  fflush (stderr);
  abort ();
#endif
}

/* Memory barrier so that GCC does not optimize out the argument.  */
#define malloc_opt_barrier(x) \
  ({ __typeof (x) __x = x; __asm ("" : "+m" (__x)); __x; })

static int
__mcheck_initialize (void (*func) (enum mcheck_status), bool in_pedantic)
{
  abortfunc = (func != NULL) ? func : &mabort;

  switch (debug_initialized)
    {
    case -1:
      /* Called before the first malloc was called.  */
      __debug_free (__debug_malloc (0));
      /* FALLTHROUGH */
    case 0:
      /* Called through the initializer hook.  */
      __malloc_debug_enable (MALLOC_MCHECK_HOOK);
      break;
    case 1:
    default:
      /* Malloc was already called.  Fail.  */
      return -1;
    }

  pedantic = in_pedantic;
  return 0;
}

static int
mcheck_usable_size (struct hdr *h)
{
  return (h - 1)->size;
}
