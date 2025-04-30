/* ABI compatibility for obsolete sigvec function from 4.2BSD.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_21)

# include <signal.h>
# include <errno.h>
# include <stddef.h>


/* These are the struct sigvec and SV_* bit definitions that
   used to be in <signal.h>.  The whole interface now exists
   solely for ABI compatibility, so it can just be here.  */
struct sigvec
  {
    __sighandler_t sv_handler;	/* Signal handler.  */
    int sv_mask;		/* Mask of signals to be blocked.  */

    int sv_flags;		/* Flags (see below).  */
  };
# define SV_ONSTACK	(1 << 0)/* Take the signal on the signal stack.  */
# define SV_INTERRUPT	(1 << 1)/* Do not restart system calls.  */
# define SV_RESETHAND	(1 << 2)/* Reset handler to SIG_DFL on receipt.  */


/* Include macros to convert between `sigset_t' and old-style mask. */
# include <sigset-cvt-mask.h>

# ifndef SA_RESETHAND
/* When sigaction lacks the extension bit for it,
   we use a wrapper handler to support SV_RESETHAND.  */
struct sigvec_wrapper_data
{
  __sighandler_t sw_handler;
  unsigned int sw_mask;
};

static void sigvec_wrapper_handler (int sig) __THROW;

static struct sigvec_wrapper_data sigvec_wrapper_data[NSIG];
# endif


/* If VEC is non-NULL, set the handler for SIG to the `sv_handler' member
   of VEC.  The signals in `sv_mask' will be blocked while the handler runs.
   If the SV_RESETHAND bit is set in `sv_flags', the handler for SIG will be
   reset to SIG_DFL before `sv_handler' is entered.  If OVEC is non-NULL,
   it is filled in with the old information for SIG.  */
int
__sigvec (int sig,
          const struct sigvec *vec,
          struct sigvec *ovec)
{
  struct sigaction old;

# ifndef SA_RESETHAND
  if (vec == NULL || !(vec->sv_flags & SV_RESETHAND))
# endif
    {
      struct sigaction new, *n;

      if (vec == NULL)
	n = NULL;
      else
	{
	  __sighandler_t handler;
	  unsigned int mask;
	  unsigned int sv_flags;
	  unsigned int sa_flags;

	  handler = vec->sv_handler;
	  mask = vec->sv_mask;
	  sv_flags = vec->sv_flags;
	  sa_flags = 0;
	  if (sv_flags & SV_ONSTACK)
	    {
# ifdef SA_ONSTACK
	      sa_flags |= SA_ONSTACK;
# else
	      __set_errno (ENOSYS);
	      return -1;
# endif
	    }
# ifdef SA_RESTART
	  if (!(sv_flags & SV_INTERRUPT))
	    sa_flags |= SA_RESTART;
# endif
# ifdef SA_RESETHAND
	  if (sv_flags & SV_RESETHAND)
	    sa_flags |= SA_RESETHAND;
# endif
	  n = &new;
	  new.sa_handler = handler;
	  if (sigset_set_old_mask (&new.sa_mask, mask) < 0)
	    return -1;
	  new.sa_flags = sa_flags;
	}

      if (__sigaction (sig, n, &old) < 0)
	return -1;
    }
# ifndef SA_RESETHAND
  else
    {
      __sighandler_t handler;
      unsigned int mask;
      struct sigvec_wrapper_data *data;
      struct sigaction wrapper;

      handler = vec->sv_handler;
      mask = (unsigned int)vec->sv_mask;
      data = &sigvec_wrapper_data[sig];
      wrapper.sa_handler = sigvec_wrapper_handler;
      /* FIXME: should we set wrapper.sa_mask, wrapper.sa_flags??  */
      data->sw_handler = handler;
      data->sw_mask = mask;

      if (__sigaction (sig, &wrapper, &old) < 0)
	return -1;
    }
# endif

  if (ovec != NULL)
    {
      __sighandler_t handler;
      unsigned int sv_flags;
      unsigned int sa_flags;
      unsigned int mask;

      handler = old.sa_handler;
      sv_flags = 0;
      sa_flags = old.sa_flags;
# ifndef SA_RESETHAND
      if (handler == sigvec_wrapper_handler)
	{
	  handler = sigvec_wrapper_data[sig].sw_handler;
	  /* should we use data->sw_mask?? */
	  sv_flags |= SV_RESETHAND;
	}
# else
     if (sa_flags & SA_RESETHAND)
	sv_flags |= SV_RESETHAND;
# endif
      mask = sigset_get_old_mask (&old.sa_mask);
# ifdef SA_ONSTACK
     if (sa_flags & SA_ONSTACK)
	sv_flags |= SV_ONSTACK;
# endif
# ifdef SA_RESTART
     if (!(sa_flags & SA_RESTART))
# endif
	sv_flags |= SV_INTERRUPT;
      ovec->sv_handler = handler;
      ovec->sv_mask = (int)mask;
      ovec->sv_flags = (int)sv_flags;
    }

  return 0;
}

compat_symbol (libc, __sigvec, sigvec, GLIBC_2_0);

# ifndef SA_RESETHAND
static void
sigvec_wrapper_handler (int sig)
{
  struct sigvec_wrapper_data *data;
  struct sigaction act;
  int save;
  __sighandler_t handler;

  data = &sigvec_wrapper_data[sig];
  act.sa_handler = SIG_DFL;
  act.sa_flags = 0;
  sigset_set_old_mask (&act.sa_mask, data->sw_mask);
  handler = data->sw_handler;
  save = errno;
  (void) __sigaction (sig, &act, (struct sigaction *) NULL);
  __set_errno (save);

  (*handler) (sig);
}
# endif  /* No SA_RESETHAND.  */

#endif  /* SHLIB_COMPAT */
