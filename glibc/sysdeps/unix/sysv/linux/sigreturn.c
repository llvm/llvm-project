/* The sigreturn syscall cannot be explicitly called on Linux, only
   implicitly by returning from a signal handler.  */
#include <signal/sigreturn.c>
