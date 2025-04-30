/* Tests for memory protection keys.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <inttypes.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>
#include <support/xsignal.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <sys/mman.h>

/* Used to force threads to wait until the main thread has set up the
   keys as intended.  */
static pthread_barrier_t barrier;

/* The keys used for testing.  These have been allocated with access
   rights set based on their array index.  */
enum { key_count = 3 };
static int keys[key_count];
static volatile int *pages[key_count];

/* Used to report results from the signal handler.  */
static volatile void *sigsegv_addr;
static volatile int sigsegv_code;
static volatile int sigsegv_pkey;
static sigjmp_buf sigsegv_jmp;

/* Used to handle expected read or write faults.  */
static void
sigsegv_handler (int signum, siginfo_t *info, void *context)
{
  sigsegv_addr = info->si_addr;
  sigsegv_code = info->si_code;
  sigsegv_pkey = info->si_pkey;
  siglongjmp (sigsegv_jmp, 2);
}

static const struct sigaction sigsegv_sigaction =
  {
    .sa_flags = SA_RESETHAND | SA_SIGINFO,
    .sa_sigaction = &sigsegv_handler,
  };

/* Check if PAGE is readable (if !WRITE) or writable (if WRITE).  */
static bool
check_page_access (int page, bool write)
{
  /* This is needed to work around bug 22396: On x86-64, siglongjmp
     does not restore the protection key access rights for the current
     thread.  We restore only the access rights for the keys under
     test.  (This is not a general solution to this problem, but it
     allows testing to proceed after a fault.)  */
  unsigned saved_rights[key_count];
  for (int i = 0; i < key_count; ++i)
    saved_rights[i] = pkey_get (keys[i]);

  volatile int *addr = pages[page];
  if (test_verbose > 0)
    {
      printf ("info: checking access at %p (page %d) for %s\n",
              addr, page, write ? "writing" : "reading");
    }
  int result = sigsetjmp (sigsegv_jmp, 1);
  if (result == 0)
    {
      xsigaction (SIGSEGV, &sigsegv_sigaction, NULL);
      if (write)
        *addr = 3;
      else
        (void) *addr;
      xsignal (SIGSEGV, SIG_DFL);
      if (test_verbose > 0)
        puts ("  --> access allowed");
      return true;
    }
  else
    {
      xsignal (SIGSEGV, SIG_DFL);
      if (test_verbose > 0)
        puts ("  --> access denied");
      TEST_COMPARE (result, 2);
      TEST_COMPARE ((uintptr_t) sigsegv_addr, (uintptr_t) addr);
      TEST_COMPARE (sigsegv_code, SEGV_PKUERR);
      TEST_COMPARE (sigsegv_pkey, keys[page]);
      for (int i = 0; i < key_count; ++i)
        TEST_COMPARE (pkey_set (keys[i], saved_rights[i]), 0);
      return false;
    }
}

static volatile sig_atomic_t sigusr1_handler_ran;
/* Used to check the behavior in signal handlers.  In x86 all access are
   revoked during signal handling.  In PowerPC the key permissions are
   inherited by the interrupted thread. This test accept both approaches.  */
static void
sigusr1_handler (int signum)
{
  TEST_COMPARE (signum, SIGUSR1);
  for (int i = 0; i < key_count; ++i)
    TEST_VERIFY (pkey_get (keys[i]) == PKEY_DISABLE_ACCESS
                 || pkey_get (keys[i]) == i);
  sigusr1_handler_ran = 1;
}

/* Used to report results from other threads.  */
struct thread_result
{
  int access_rights[key_count];
  pthread_t next_thread;
};

/* Return the thread's access rights for the keys under test.  */
static void *
get_thread_func (void *closure)
{
  struct thread_result *result = xmalloc (sizeof (*result));
  for (int i = 0; i < key_count; ++i)
    result->access_rights[i] = pkey_get (keys[i]);
  memset (&result->next_thread, 0, sizeof (result->next_thread));
  return result;
}

/* Wait for initialization and then check that the current thread does
   not have access through the keys under test.  */
static void *
delayed_thread_func (void *closure)
{
  bool check_access = *(bool *) closure;
  pthread_barrier_wait (&barrier);
  struct thread_result *result = get_thread_func (NULL);

  if (check_access)
    {
      /* Also check directly.  This code should not run with other
         threads in parallel because of the SIGSEGV handler which is
         installed by check_page_access.  */
      for (int i = 0; i < key_count; ++i)
        {
          TEST_VERIFY (!check_page_access (i, false));
          TEST_VERIFY (!check_page_access (i, true));
        }
    }

  result->next_thread = xpthread_create (NULL, get_thread_func, NULL);
  return result;
}

static int
do_test (void)
{
  long pagesize = xsysconf (_SC_PAGESIZE);

  /* pkey_mprotect with key -1 should work even when there is no
     protection key support.  */
  {
    int *page = xmmap (NULL, pagesize, PROT_NONE,
                       MAP_ANONYMOUS | MAP_PRIVATE, -1);
    TEST_COMPARE (pkey_mprotect (page, pagesize, PROT_READ | PROT_WRITE, -1),
                  0);
    volatile int *vpage = page;
    *vpage = 5;
    TEST_COMPARE (*vpage, 5);
    xmunmap (page, pagesize);
  }

  xpthread_barrier_init (&barrier, NULL, 2);
  bool delayed_thread_check_access = true;
  pthread_t delayed_thread = xpthread_create
    (NULL, &delayed_thread_func, &delayed_thread_check_access);

  keys[0] = pkey_alloc (0, 0);
  if (keys[0] < 0)
    {
      if (errno == ENOSYS)
        FAIL_UNSUPPORTED
          ("kernel does not support memory protection keys");
      if (errno == EINVAL)
        FAIL_UNSUPPORTED
          ("CPU does not support memory protection keys: %m");
      if (errno == ENOSPC)
        FAIL_UNSUPPORTED
          ("no keys available or kernel does not support memory"
           " protection keys");
      FAIL_EXIT1 ("pkey_alloc: %m");
    }
  TEST_COMPARE (pkey_get (keys[0]), 0);
  for (int i = 1; i < key_count; ++i)
    {
      keys[i] = pkey_alloc (0, i);
      if (keys[i] < 0)
        FAIL_EXIT1 ("pkey_alloc (0, %d): %m", i);
      /* pkey_alloc is supposed to change the current thread's access
         rights for the new key.  */
      TEST_COMPARE (pkey_get (keys[i]), i);
    }
  /* Check that all the keys have the expected access rights for the
     current thread.  */
  for (int i = 0; i < key_count; ++i)
    TEST_COMPARE (pkey_get (keys[i]), i);

  /* Allocate a test page for each key.  */
  for (int i = 0; i < key_count; ++i)
    {
      pages[i] = xmmap (NULL, pagesize, PROT_READ | PROT_WRITE,
                        MAP_ANONYMOUS | MAP_PRIVATE, -1);
      TEST_COMPARE (pkey_mprotect ((void *) pages[i], pagesize,
                                   PROT_READ | PROT_WRITE, keys[i]), 0);
    }

  /* Check that the initial thread does not have access to the new
     keys.  */
  {
    pthread_barrier_wait (&barrier);
    struct thread_result *result = xpthread_join (delayed_thread);
    for (int i = 0; i < key_count; ++i)
      TEST_COMPARE (result->access_rights[i],
                    PKEY_DISABLE_ACCESS);
    struct thread_result *result2 = xpthread_join (result->next_thread);
    for (int i = 0; i < key_count; ++i)
      TEST_COMPARE (result->access_rights[i],
                    PKEY_DISABLE_ACCESS);
    free (result);
    free (result2);
  }

  /* Check that the current thread access rights are inherited by new
     threads.  */
  {
    pthread_t get_thread = xpthread_create (NULL, get_thread_func, NULL);
    struct thread_result *result = xpthread_join (get_thread);
    for (int i = 0; i < key_count; ++i)
      TEST_COMPARE (result->access_rights[i], i);
    free (result);
  }

  for (int i = 0; i < key_count; ++i)
    TEST_COMPARE (pkey_get (keys[i]), i);

  /* Check that in a signal handler, there is no access.  */
  xsignal (SIGUSR1, &sigusr1_handler);
  xraise (SIGUSR1);
  xsignal (SIGUSR1, SIG_DFL);
  TEST_COMPARE (sigusr1_handler_ran, 1);

  /* The first key results in a writable page.  */
  TEST_VERIFY (check_page_access (0, false));
  TEST_VERIFY (check_page_access (0, true));

  /* The other keys do not.   */
  for (int i = 1; i < key_count; ++i)
    {
      if (test_verbose)
        printf ("info: checking access for key %d, bits 0x%x\n",
                i, pkey_get (keys[i]));
      for (int j = 0; j < key_count; ++j)
        TEST_COMPARE (pkey_get (keys[j]), j);
      if (i & PKEY_DISABLE_ACCESS)
        {
          TEST_VERIFY (!check_page_access (i, false));
          TEST_VERIFY (!check_page_access (i, true));
        }
      else
        {
          TEST_VERIFY (i & PKEY_DISABLE_WRITE);
          TEST_VERIFY (check_page_access (i, false));
          TEST_VERIFY (!check_page_access (i, true));
        }
    }

  /* But if we set the current thread's access rights, we gain
     access.  */
  for (int do_write = 0; do_write < 2; ++do_write)
    for (int allowed_key = 0; allowed_key < key_count; ++allowed_key)
      {
        for (int i = 0; i < key_count; ++i)
          if (i == allowed_key)
            {
              if (do_write)
                TEST_COMPARE (pkey_set (keys[i], 0), 0);
              else
                TEST_COMPARE (pkey_set (keys[i], PKEY_DISABLE_WRITE), 0);
            }
          else
            TEST_COMPARE (pkey_set (keys[i], PKEY_DISABLE_ACCESS), 0);

        if (test_verbose)
          printf ("info: key %d is allowed access for %s\n",
                  allowed_key, do_write ? "writing" : "reading");
        for (int i = 0; i < key_count; ++i)
          if (i == allowed_key)
            {
              TEST_VERIFY (check_page_access (i, false));
              TEST_VERIFY (check_page_access (i, true) == do_write);
            }
          else
            {
              TEST_VERIFY (!check_page_access (i, false));
              TEST_VERIFY (!check_page_access (i, true));
            }
      }

  /* Restore access to all keys, and launch a thread which should
     inherit that access.  */
  for (int i = 0; i < key_count; ++i)
    {
      TEST_COMPARE (pkey_set (keys[i], 0), 0);
      TEST_VERIFY (check_page_access (i, false));
      TEST_VERIFY (check_page_access (i, true));
    }
  delayed_thread_check_access = false;
  delayed_thread = xpthread_create
    (NULL, delayed_thread_func, &delayed_thread_check_access);

  TEST_COMPARE (pkey_free (keys[0]), 0);
  /* Second pkey_free will fail because the key has already been
     freed.  */
  TEST_COMPARE (pkey_free (keys[0]),-1);
  TEST_COMPARE (errno, EINVAL);
  for (int i = 1; i < key_count; ++i)
    TEST_COMPARE (pkey_free (keys[i]), 0);

  /* Check what happens to running threads which have access to
     previously allocated protection keys.  The implemented behavior
     is somewhat dubious: Ideally, pkey_free should revoke access to
     that key and pkey_alloc of the same (numeric) key should not
     implicitly confer access to already-running threads, but this is
     not what happens in practice.  */
  {
    /* The limit is in place to avoid running indefinitely in case
       there many keys available.  */
    int *keys_array = xcalloc (100000, sizeof (*keys_array));
    int keys_allocated = 0;
    while (keys_allocated < 100000)
      {
        int new_key = pkey_alloc (0, PKEY_DISABLE_WRITE);
        if (new_key < 0)
          {
            /* No key reuse observed before running out of keys.  */
            TEST_COMPARE (errno, ENOSPC);
            break;
          }
        for (int i = 0; i < key_count; ++i)
          if (new_key == keys[i])
            {
              /* We allocated the key with disabled write access.
                 This should affect the protection state of the
                 existing page.  */
              TEST_VERIFY (check_page_access (i, false));
              TEST_VERIFY (!check_page_access (i, true));

              xpthread_barrier_wait (&barrier);
              struct thread_result *result = xpthread_join (delayed_thread);
              /* The thread which was launched before should still have
                 access to the key.  */
              TEST_COMPARE (result->access_rights[i], 0);
              struct thread_result *result2
                = xpthread_join (result->next_thread);
              /* Same for a thread which is launched afterwards from
                 the old thread.  */
              TEST_COMPARE (result2->access_rights[i], 0);
              free (result);
              free (result2);
              keys_array[keys_allocated++] = new_key;
              goto after_key_search;
            }
        /* Save key for later deallocation.  */
        keys_array[keys_allocated++] = new_key;
      }
  after_key_search:
    /* Deallocate the keys allocated for testing purposes.  */
    for (int j = 0; j < keys_allocated; ++j)
      TEST_COMPARE (pkey_free (keys_array[j]), 0);
    free (keys_array);
  }

  for (int i = 0; i < key_count; ++i)
    xmunmap ((void *) pages[i], pagesize);

  xpthread_barrier_destroy (&barrier);
  return 0;
}

#include <support/test-driver.c>
