/* Test for i386 sigaction sa_restorer handling (BZ#21269)
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

/* This is based on Linux test tools/testing/selftests/x86/ldt_gdt.c,
   more specifically in do_multicpu_tests function.  The main changes
   are:

   - C11 atomics instead of plain access.
   - Remove x86_64 support which simplifies the syscall handling
     and fallbacks.
   - Replicate only the test required to trigger the issue for the
     BZ#21269.  */

#include <stdatomic.h>

#include <asm/ldt.h>
#include <linux/futex.h>

#include <setjmp.h>
#include <signal.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/mman.h>

#include <support/xunistd.h>
#include <support/check.h>
#include <support/xthread.h>

static int
xset_thread_area (struct user_desc *u_info)
{
  long ret = syscall (SYS_set_thread_area, u_info);
  TEST_VERIFY_EXIT (ret == 0);
  return ret;
}

static void
xmodify_ldt (int func, const void *ptr, unsigned long bytecount)
{
  TEST_VERIFY_EXIT (syscall (SYS_modify_ldt, 1, ptr, bytecount) == 0);
}

static int
futex (int *uaddr, int futex_op, int val, void *timeout, int *uaddr2,
	int val3)
{
  return syscall (SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}

static void
xsethandler (int sig, void (*handler)(int, siginfo_t *, void *), int flags)
{
  struct sigaction sa = { 0 };
  sa.sa_sigaction = handler;
  sa.sa_flags = SA_SIGINFO | flags;
  TEST_VERIFY_EXIT (sigemptyset (&sa.sa_mask) == 0);
  TEST_VERIFY_EXIT (sigaction (sig, &sa, 0) == 0);
}

static jmp_buf jmpbuf;

static void
sigsegv_handler (int sig, siginfo_t *info, void *ctx_void)
{
  siglongjmp (jmpbuf, 1);
}

/* Points to an array of 1024 ints, each holding its own index.  */
static const unsigned int *counter_page;
static struct user_desc *low_user_desc;
static struct user_desc *low_user_desc_clear; /* Used to delete GDT entry.  */
static int gdt_entry_num;

static void
setup_counter_page (void)
{
  long page_size = sysconf (_SC_PAGE_SIZE);
  TEST_VERIFY_EXIT (page_size > 0);
  unsigned int *page = xmmap (NULL, page_size, PROT_READ | PROT_WRITE,
			      MAP_ANONYMOUS | MAP_PRIVATE | MAP_32BIT, -1);
  for (int i = 0; i < (page_size / sizeof (unsigned int)); i++)
    page[i] = i;
  counter_page = page;
}

static void
setup_low_user_desc (void)
{
  low_user_desc = xmmap (NULL, 2 * sizeof (struct user_desc),
			 PROT_READ | PROT_WRITE,
			 MAP_ANONYMOUS | MAP_PRIVATE | MAP_32BIT, -1);

  low_user_desc->entry_number    = -1;
  low_user_desc->base_addr       = (unsigned long) &counter_page[1];
  low_user_desc->limit           = 0xffff;
  low_user_desc->seg_32bit       = 1;
  low_user_desc->contents        = 0;
  low_user_desc->read_exec_only  = 0;
  low_user_desc->limit_in_pages  = 1;
  low_user_desc->seg_not_present = 0;
  low_user_desc->useable         = 0;

  xset_thread_area (low_user_desc);

  low_user_desc_clear = low_user_desc + 1;
  low_user_desc_clear->entry_number = gdt_entry_num;
  low_user_desc_clear->read_exec_only = 1;
  low_user_desc_clear->seg_not_present = 1;
}

/* Possible values of futex:
   0: thread is idle.
   1: thread armed.
   2: thread should clear LDT entry 0.
   3: thread should exit.  */
static atomic_uint ftx;

static void *
threadproc (void *ctx)
{
  while (1)
    {
      futex ((int *) &ftx, FUTEX_WAIT, 1, NULL, NULL, 0);
      while (atomic_load (&ftx) != 2)
	{
	  if (atomic_load (&ftx) >= 3)
	    return NULL;
	}

      /* clear LDT entry 0.  */
      const struct user_desc desc = { 0 };
      xmodify_ldt (1, &desc, sizeof (desc));

      /* If ftx == 2, set it to zero,  If ftx == 100, quit.  */
      if (atomic_fetch_add (&ftx, -2) != 2)
	return NULL;
    }
}


/* As described in testcase, for historical reasons x86_32 Linux (and compat
   on x86_64) interprets SA_RESTORER clear with nonzero sa_restorer as a
   request for stack switching if the SS segment is 'funny' (this is default
   scenario for vDSO system).  This means that anything that tries to mix
   signal handling with segmentation should explicit clear the sa_restorer.

   This testcase check if sigaction in fact does it by changing the local
   descriptor table (LDT) through the modify_ldt syscall and triggering
   a synchronous segfault on iret fault by trying to install an invalid
   segment.  With a correct zeroed sa_restorer it should not trigger an
   'real' SEGSEGV and allows the siglongjmp in signal handler.  */

static int
do_test (void)
{
  setup_counter_page ();
  setup_low_user_desc ();

  pthread_t thread;
  unsigned short orig_ss;

  xsethandler (SIGSEGV, sigsegv_handler, 0);
  /* 32-bit kernels send SIGILL instead of SIGSEGV on IRET faults.  */
  xsethandler (SIGILL, sigsegv_handler, 0);
  /* Some kernels send SIGBUS instead.  */
  xsethandler (SIGBUS, sigsegv_handler, 0);

  thread = xpthread_create (0, threadproc, 0);

  asm volatile ("mov %%ss, %0" : "=rm" (orig_ss));

  for (int i = 0; i < 5; i++)
    {
      if (sigsetjmp (jmpbuf, 1) != 0)
	continue;

      /* Make sure the thread is ready after the last test. */
      while (atomic_load (&ftx) != 0)
	;

      struct user_desc desc = {
	.entry_number       = 0,
	.base_addr          = 0,
	.limit              = 0xffff,
	.seg_32bit          = 1,
	.contents           = 0,
	.read_exec_only     = 0,
	.limit_in_pages     = 1,
	.seg_not_present    = 0,
	.useable            = 0
      };

      xmodify_ldt (0x11, &desc, sizeof (desc));

      /* Arm the thread.  */
      ftx = 1;
      futex ((int*) &ftx, FUTEX_WAKE, 0, NULL, NULL, 0);

      asm volatile ("mov %0, %%ss" : : "r" (0x7));

      /* Fire up thread modify_ldt call.  */
      atomic_store (&ftx, 2);

      while (atomic_load (&ftx) != 0)
	;

      /* On success, modify_ldt will segfault us synchronously and we will
	 escape via siglongjmp.  */
      support_record_failure ();
    }

  atomic_store (&ftx, 100);
  futex ((int*) &ftx, FUTEX_WAKE, 0, NULL, NULL, 0);

  xpthread_join (thread);

  return 0;
}

#include <support/test-driver.c>
