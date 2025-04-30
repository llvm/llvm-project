#ifndef __sigevent_t_defined
#define __sigevent_t_defined 1

#include <bits/types.h>
#include <bits/types/__sigval_t.h>

/* Structure to transport application-defined values with signals.  */
typedef struct sigevent
  {
    __sigval_t sigev_value;
    int sigev_signo;
    int sigev_notify;
    void (*sigev_notify_function) (__sigval_t);	    /* Function to start.  */
    void *sigev_notify_attributes;		    /* Really pthread_attr_t.*/
  } sigevent_t;

#endif
