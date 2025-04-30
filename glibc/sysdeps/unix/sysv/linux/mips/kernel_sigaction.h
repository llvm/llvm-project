#ifndef _KERNEL_SIGACTION_H
# define _KERNEL_SIGACTION_H

/* This is the sigaction structure from the Linux 3.2 kernel.  */
struct kernel_sigaction
{
  unsigned int    sa_flags;
  __sighandler_t  k_sa_handler;
  sigset_t        sa_mask;
};

#endif
