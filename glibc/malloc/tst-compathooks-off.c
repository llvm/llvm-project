/* Minimal tests to verify libc_malloc_debug.so functionality.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <shlib-compat.h>
#include <libc-diag.h>

#include <support/check.h>
#include <support/support.h>

extern void (*volatile __free_hook) (void *, const void *);
extern void *(*volatile __malloc_hook)(size_t, const void *);
extern void *(*volatile __realloc_hook)(void *, size_t, const void *);
extern void *(*volatile __memalign_hook)(size_t, size_t, const void *);

int hook_count, call_count;

DIAG_PUSH_NEEDS_COMMENT;
DIAG_IGNORE_NEEDS_COMMENT (4.9, "-Wdeprecated-declarations");

void
free_called (void *mem, const void *address)
{
  hook_count++;
  __free_hook = NULL;
  free (mem);
  __free_hook = free_called;
}

void *
malloc_called (size_t bytes, const void *address)
{
  hook_count++;
  __malloc_hook = NULL;
  void *mem = malloc (bytes);
  __malloc_hook = malloc_called;
  return mem;
}

void *
realloc_called (void *oldptr, size_t bytes, const void *address)
{
  hook_count++;
  __realloc_hook = NULL;
  void *mem = realloc (oldptr, bytes);
  __realloc_hook = realloc_called;
  return mem;
}

void *
calloc_called (size_t n, size_t size, const void *address)
{
  hook_count++;
  __malloc_hook = NULL;
  void *mem = calloc (n, size);
  __malloc_hook = malloc_called;
  return mem;
}

void *
memalign_called (size_t align, size_t size, const void *address)
{
  hook_count++;
  __memalign_hook = NULL;
  void *mem = memalign (align, size);
  __memalign_hook = memalign_called;
  return mem;
}

static void initialize_hooks (void)
{
  __free_hook = free_called;
  __malloc_hook = malloc_called;
  __realloc_hook = realloc_called;
  __memalign_hook = memalign_called;
}
void (*__malloc_initialize_hook) (void) = initialize_hooks;
compat_symbol_reference (libc, __malloc_initialize_hook,
			 __malloc_initialize_hook, GLIBC_2_0);
compat_symbol_reference (libc, __free_hook,
			 __free_hook, GLIBC_2_0);
compat_symbol_reference (libc, __malloc_hook,
			 __malloc_hook, GLIBC_2_0);
compat_symbol_reference (libc, __realloc_hook,
			 __realloc_hook, GLIBC_2_0);
compat_symbol_reference (libc, __memalign_hook,
			 __memalign_hook, GLIBC_2_0);

DIAG_POP_NEEDS_COMMENT;

static int
do_test (void)
{
  void *p;
  p = malloc (0);
  TEST_VERIFY_EXIT (p != NULL);
  call_count++;

  p = realloc (p, 0);
  TEST_VERIFY_EXIT (p == NULL);
  call_count++;

  p = calloc (512, 1);
  TEST_VERIFY_EXIT (p != NULL);
  call_count++;

  free (p);
  call_count++;

  p = memalign (0x100, 0x100);
  TEST_VERIFY_EXIT (p != NULL);
  call_count++;

  free (p);
  call_count++;

  printf ("call_count: %d, hook_count: %d\n", call_count, hook_count);

#ifdef HOOKS_ENABLED
  TEST_VERIFY_EXIT (call_count == hook_count);
#else
  TEST_VERIFY_EXIT (hook_count == 0);
#endif

  exit (0);
}

#include <support/test-driver.c>
