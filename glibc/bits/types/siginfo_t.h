#ifndef __siginfo_t_defined
#define __siginfo_t_defined 1

#include <bits/types.h>
#include <bits/types/__sigval_t.h>

typedef struct
  {
    int si_signo;		/* Signal number.  */
    int si_errno;		/* If non-zero, an errno value associated with
				   this signal, as defined in <errno.h>.  */
    int si_code;		/* Signal code.  */
    __pid_t si_pid;		/* Sending process ID.  */
    __uid_t si_uid;		/* Real user ID of sending process.  */
    void *si_addr;		/* Address of faulting instruction.  */
    int si_status;		/* Exit value or signal.  */
    long int si_band;		/* Band event for SIGPOLL.  */
    __sigval_t si_value;	/* Signal value.  */
  } siginfo_t;

#endif
