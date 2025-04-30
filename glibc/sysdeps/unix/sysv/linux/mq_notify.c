/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contribute by Ulrich Drepper <drepper@redhat.com>, 2004.

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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sysdep.h>
#include <unistd.h>
#include <sys/socket.h>
#include <not-cancel.h>
#include <pthreadP.h>
#include <shlib-compat.h>

/* Defined in the kernel headers: */
#define NOTIFY_COOKIE_LEN	32	/* Length of the cookie used.  */
#define NOTIFY_WOKENUP		1	/* Code for notifcation.  */
#define NOTIFY_REMOVED		2	/* Code for closed message queue
					   of de-notifcation.  */


/* Data structure for the queued notification requests.  */
union notify_data
{
  struct
  {
    void (*fct) (union sigval);	/* The function to run.  */
    union sigval param;		/* The parameter to pass.  */
    pthread_attr_t *attr;	/* Attributes to create the thread with.  */
    /* NB: on 64-bit machines the struct as a size of 24 bytes.  Which means
       byte 31 can still be used for returning the status.  */
  };
  char raw[NOTIFY_COOKIE_LEN];
};


/* Keep track of the initialization.  */
static pthread_once_t once = PTHREAD_ONCE_INIT;


/* The netlink socket.  */
static int netlink_socket = -1;


/* Barrier used to make sure data passed to the new thread is not
   resused by the parent.  */
static pthread_barrier_t notify_barrier;


/* Modify the signal mask.  We move this into a separate function so
   that the stack space needed for sigset_t is not deducted from what
   the thread can use.  */
static int
__attribute__ ((noinline))
change_sigmask (int how, sigset_t *oss)
{
  sigset_t ss;
  sigfillset (&ss);
  return __pthread_sigmask (how, &ss, oss);
}


/* The function used for the notification.  */
static void *
notification_function (void *arg)
{
  /* Copy the function and parameter so that the parent thread can go
     on with its life.  */
  volatile union notify_data *data = (volatile union notify_data *) arg;
  void (*fct) (union sigval) = data->fct;
  union sigval param = data->param;

  /* Let the parent go.  */
  (void) __pthread_barrier_wait (&notify_barrier);

  /* Make the thread detached.  */
  __pthread_detach (__pthread_self ());

  /* The parent thread has all signals blocked.  This is probably a
     bit surprising for this thread.  So we unblock all of them.  */
  (void) change_sigmask (SIG_UNBLOCK, NULL);

  /* Now run the user code.  */
  fct (param);

  /* And we are done.  */
  return NULL;
}


/* Helper thread.  */
static void *
helper_thread (void *arg)
{
  while (1)
    {
      union notify_data data;

      ssize_t n = __recv (netlink_socket, &data, sizeof (data),
			  MSG_NOSIGNAL | MSG_WAITALL);
      if (n < NOTIFY_COOKIE_LEN)
	continue;

      if (data.raw[NOTIFY_COOKIE_LEN - 1] == NOTIFY_WOKENUP)
	{
	  /* Just create the thread as instructed.  There is no way to
	     report a problem with creating a thread.  */
	  pthread_t th;
	  if (__pthread_create (&th, data.attr, notification_function, &data)
	      == 0)
	    /* Since we passed a pointer to DATA to the new thread we have
	       to wait until it is done with it.  */
	    (void) __pthread_barrier_wait (&notify_barrier);
	}
      else if (data.raw[NOTIFY_COOKIE_LEN - 1] == NOTIFY_REMOVED)
	{
	  /* The only state we keep is the copy of the thread attributes.  */
	  __pthread_attr_destroy (data.attr);
	  free (data.attr);
	}
    }
  return NULL;
}


void
__mq_notify_fork_subprocess (void)
{
  once = PTHREAD_ONCE_INIT;
}


static void
init_mq_netlink (void)
{
  /* This code might be called a second time after fork().  The file
     descriptor is inherited from the parent.  */
  if (netlink_socket == -1)
    {
      /* Just a normal netlink socket, not bound.  */
      netlink_socket = __socket (AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, 0);
      /* No need to do more if we have no socket.  */
      if (netlink_socket == -1)
	return;
    }

  int err = 1;

  /* Initialize the barrier.  */
  if (__pthread_barrier_init (&notify_barrier, NULL, 2) == 0)
    {
      /* Create the helper thread.  */
      pthread_attr_t attr;
      __pthread_attr_init (&attr);
      __pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
      /* We do not need much stack space, the bare minimum will be enough.  */
      __pthread_attr_setstacksize (&attr, __pthread_get_minstack (&attr));

      /* Temporarily block all signals so that the newly created
	 thread inherits the mask.  */
      sigset_t oss;
      int have_no_oss = change_sigmask (SIG_BLOCK, &oss);

      pthread_t th;
      err = __pthread_create (&th, &attr, helper_thread, NULL);

      /* Reset the signal mask.  */
      if (!have_no_oss)
	__pthread_sigmask (SIG_SETMASK, &oss, NULL);

      __pthread_attr_destroy (&attr);
    }

  if (err != 0)
    {
      __close_nocancel_nostatus (netlink_socket);
      netlink_socket = -1;
    }
}


/* Register notification upon message arrival to an empty message queue
   MQDES.  */
int
__mq_notify (mqd_t mqdes, const struct sigevent *notification)
{
  /* Make sure the type is correctly defined.  */
  assert (sizeof (union notify_data) == NOTIFY_COOKIE_LEN);

  /* Special treatment needed for SIGEV_THREAD.  */
  if (notification == NULL || notification->sigev_notify != SIGEV_THREAD)
    return INLINE_SYSCALL (mq_notify, 2, mqdes, notification);

  /* The kernel cannot directly start threads.  This will have to be
     done at userlevel.  Since we cannot start threads from signal
     handlers we have to create a dedicated thread which waits for
     notifications for arriving messages and creates threads in
     response.  */

  /* Initialize only once.  */
  __pthread_once (&once, init_mq_netlink);

  /* If we cannot create the netlink socket we cannot provide
     SIGEV_THREAD support.  */
  if (__glibc_unlikely (netlink_socket == -1))
    {
      __set_errno (ENOSYS);
      return -1;
    }

  /* Create the cookie.  It will hold almost all the state.  */
  union notify_data data;
  memset (&data, '\0', sizeof (data));
  data.fct = notification->sigev_notify_function;
  data.param = notification->sigev_value;

  if (notification->sigev_notify_attributes != NULL)
    {
      /* The thread attribute has to be allocated separately.  */
      data.attr = (pthread_attr_t *) malloc (sizeof (pthread_attr_t));
      if (data.attr == NULL)
	return -1;

      int ret = __pthread_attr_copy (data.attr,
				     notification->sigev_notify_attributes);
      if (ret != 0)
	{
	  free (data.attr);
	  __set_errno (ret);
	  return -1;
	}
    }

  /* Construct the new request.  */
  struct sigevent se;
  se.sigev_notify = SIGEV_THREAD;
  se.sigev_signo = netlink_socket;
  se.sigev_value.sival_ptr = &data;

  /* Tell the kernel.  */
  int retval = INLINE_SYSCALL (mq_notify, 2, mqdes, &se);

  /* If it failed, free the allocated memory.  */
  if (retval != 0 && data.attr != NULL)
    {
      __pthread_attr_destroy (data.attr);
      free (data.attr);
    }

  return retval;
}
versioned_symbol (libc, __mq_notify, mq_notify, GLIBC_2_34);
libc_hidden_ver (__mq_notify, mq_notify)
#if OTHER_SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (librt, __mq_notify, mq_notify, GLIBC_2_3_4);
#endif
