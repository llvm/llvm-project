/* __sigset_t manipulators.  Generic/BSD version.
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

#ifndef _SIGSETOPS_H
#define _SIGSETOPS_H 1

#include <signal.h>

/* Return a mask that includes SIG only.  The cast to `sigset_t' avoids
   overflow if `sigset_t' is wider than `int'.  */
# define __sigmask(sig) (((__sigset_t) 1) << ((sig) - 1))

#define __sigemptyset(set)			\
  (__extension__ ({				\
    *(set) = (__sigset_t) 0;			\
    0;						\
  }))
#define __sigfillset(set)			\
  (__extension__ ({				\
    *(set) = ~(__sigset_t) 0;			\
    0;						\
  }))

# define __sigisemptyset(set)			\
  (*(set) == (__sigset_t) 0)

# define __sigandset(dest, left, right)		\
  (__extension__ ({				\
    *(dest) = *(left) & *(right);		\
    0;						\
  }))

# define __sigorset(dest, left, right)		\
  (__extension__ ({				\
    *(dest) = *(left) | *(right);		\
    0;						\
  }))

/* These macros needn't check for a bogus signal number;
   checking is done in the non-__ versions.  */
# define __sigismember(set, sig)		\
  (__extension__ ({				\
    __sigset_t __mask = __sigmask (sig);	\
    *(set) & __mask ? 1 : 0;			\
  }))

# define __sigaddset(set, sig)			\
  (__extension__ ({				\
    __sigset_t __mask = __sigmask (sig);	\
    *(set) |= __mask;				\
    0;						\
  }))

# define __sigdelset(set, sig)			\
  (__extension__ ({				\
    __sigset_t __mask = __sigmask (sig);	\
    *(set) &= ~__mask;				\
    0;						\
  }))

#endif
