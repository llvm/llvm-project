/* Low-level statistical profiling support function.  Mach/Hurd version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <hurd.h>
#include <mach/mach4.h>
#include <mach/pc_sample.h>
#include <lock-intern.h>
#include <assert.h>
#include <libc-internal.h>


#define MAX_PC_SAMPLES	512	/* XXX ought to be exported in kernel hdr */

static thread_t profile_thread = MACH_PORT_NULL;
static u_short *samples;
static size_t maxsamples;
static size_t pc_offset;
static size_t sample_scale;
static sampled_pc_seqno_t seqno;
static spin_lock_t lock = SPIN_LOCK_INITIALIZER;
static mach_msg_timeout_t collector_timeout; /* ms between collections.  */
static int profile_tick;

/* Reply port used by profiler thread */
static mach_port_t profil_reply_port = MACH_PORT_NULL;

/* Forwards */
static kern_return_t profil_task_get_sampled_pcs (mach_port_t,
						  sampled_pc_seqno_t *,
						  sampled_pc_array_t,
						  mach_msg_type_number_t *);
static void fetch_samples (void);
static void profile_waiter (void);

/* Enable statistical profiling, writing samples of the PC into at most
   SIZE bytes of SAMPLE_BUFFER; every processor clock tick while profiling
   is enabled, the system examines the user PC and increments
   SAMPLE_BUFFER[((PC - OFFSET) / 2) * SCALE / 65536].  If SCALE is zero,
   disable profiling.  Returns zero on success, -1 on error.  */

static error_t
update_waiter (u_short *sample_buffer, size_t size, size_t offset, u_int scale)
{
  error_t err;

  if (profile_thread == MACH_PORT_NULL)
    {
      if (profil_reply_port == MACH_PORT_NULL)
	profil_reply_port = __mach_reply_port ();
      /* Set up the profiling collector thread.  */
      err = __thread_create (__mach_task_self (), &profile_thread);
      if (! err)
	err = __mach_setup_thread (__mach_task_self (), profile_thread,
				   &profile_waiter, NULL, NULL);
      if (! err)
	err = __mach_setup_tls(profile_thread);
    }
  else
    err = 0;

  if (! err)
    {
      err = __task_enable_pc_sampling (__mach_task_self (), &profile_tick,
				       SAMPLED_PC_PERIODIC);
      if (!err && sample_scale == 0)
	/* Profiling was not turned on, so the collector thread was
	   suspended.  Resume it.  */
	err = __thread_resume (profile_thread);
      if (! err)
	{
	  samples = sample_buffer;
	  maxsamples = size / sizeof *sample_buffer;
	  pc_offset = offset;
	  sample_scale = scale;
	  /* Calculate a good period for the collector thread.  From TICK
	     and the kernel buffer size we get the length of time it takes
	     to fill the buffer; translate that to milliseconds for
	     mach_msg, and chop it in half for general lag factor.  */
	  collector_timeout = MAX_PC_SAMPLES * profile_tick / 1000 / 2;
	}
    }

  return err;
}

int
__profile_frequency (void)
{
  return 1000000 / profile_tick;
}
libc_hidden_def (__profile_frequency)

int
__profil (u_short *sample_buffer, size_t size, size_t offset, u_int scale)
{
  error_t err;

  __spin_lock (&lock);

  if (scale == 0)
    {
      /* Disable profiling.  */
      int count;

      if (profile_thread != MACH_PORT_NULL)
	__thread_suspend (profile_thread);

      /* Fetch the last set of samples */
      if (sample_scale)
	fetch_samples ();

      err = __task_disable_pc_sampling (__mach_task_self (), &count);
      sample_scale = 0;
      seqno = 0;
    }
  else
    err = update_waiter (sample_buffer, size, offset, scale);

  __spin_unlock (&lock);

  return err ? __hurd_fail (err) : 0;
}
weak_alias (__profil, profil)

static volatile error_t special_profil_failure;

/* Fetch PC samples.  This function must be very careful not to depend
   on Hurd TLS variables.  We arrange that by using a special
   stub arranged for at the end of this file. */
static void
fetch_samples (void)
{
  sampled_pc_t pc_samples[MAX_PC_SAMPLES];
  mach_msg_type_number_t nsamples, i;
  error_t err;

  nsamples = MAX_PC_SAMPLES;

  err = profil_task_get_sampled_pcs (__mach_task_self (), &seqno,
				     pc_samples, &nsamples);
  if (err)
    {
      static volatile int a, b;

      special_profil_failure = err;
      a = 1;
      b = 0;
      while (1)
	a = a / b;
    }

  for (i = 0; i < nsamples; ++i)
    {
      /* Do arithmetic in long long to avoid overflow problems. */
      long long pc_difference = pc_samples[i].pc - pc_offset;
      size_t idx = ((pc_difference / 2) * sample_scale) / 65536;
      if (idx < maxsamples)
	++samples[idx];
    }
}


/* This function must be very careful not to depend on Hurd TLS
   variables.  We arrange that by using special stubs arranged for at the
   end of this file. */
static void
profile_waiter (void)
{
  mach_msg_header_t msg;
  mach_port_t timeout_reply_port;

  timeout_reply_port = __mach_reply_port ();

  while (1)
    {
      __spin_lock (&lock);

      fetch_samples ();

      __spin_unlock (&lock);

      __mach_msg (&msg, MACH_RCV_MSG|MACH_RCV_TIMEOUT, 0, sizeof msg,
		  timeout_reply_port, collector_timeout, MACH_PORT_NULL);
    }
}

/* Fork interaction */

/* Before fork, lock the interlock so that we are in a clean state. */
static void
fork_profil_prepare (void)
{
  __spin_lock (&lock);
}
text_set_element (_hurd_fork_prepare_hook, fork_profil_prepare);

/* In the parent, unlock the interlock once fork is complete. */
static void
fork_profil_parent (void)
{
  __spin_unlock (&lock);
}
text_set_element (_hurd_fork_parent_hook, fork_profil_parent);

/* In the child, unlock the interlock, and start a profiling thread up
   if necessary. */
static void
fork_profil_child (void)
{
  u_short *sb;
  size_t n, o, ss;
  error_t err;

  __spin_unlock (&lock);

  if (profile_thread != MACH_PORT_NULL)
    {
      __mach_port_deallocate (__mach_task_self (), profile_thread);
      profile_thread = MACH_PORT_NULL;
    }

  sb = samples;
  samples = NULL;
  n = maxsamples;
  maxsamples = 0;
  o = pc_offset;
  pc_offset = 0;
  ss = sample_scale;
  sample_scale = 0;

  if (ss != 0)
    {
      err = update_waiter (sb, n * sizeof *sb, o, ss);
      assert_perror (err);
    }
}
text_set_element (_hurd_fork_child_hook, fork_profil_child);




/* Special RPC stubs for profile_waiter are made by including the normal
   source code, with special CPP state to prevent it from doing the
   usual thing. */

/* Include these first; then our #define's will take full effect, not
   being overridden. */
#include <mach/mig_support.h>

/* This need not do anything; it is always associated with errors, which
   are fatal in profile_waiter anyhow. */
#define __mig_put_reply_port(foo)

/* Use our static variable instead of the usual TLS mechanism for
   this. */
#define __mig_get_reply_port() profil_reply_port

/* Make the functions show up as static */
#define mig_external static

/* Turn off the attempt to generate ld aliasing records. */
#undef weak_alias
#define weak_alias(a,b)

/* And change their names to avoid confusing disasters. */
#define __vm_deallocate_rpc profil_vm_deallocate
#define __task_get_sampled_pcs profil_task_get_sampled_pcs

/* And include the source code */
#include <../mach/RPC_task_get_sampled_pcs.c>
