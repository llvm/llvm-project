#define NOCANCEL
#define __libc_fcntl __fcntl_nocancel
#include <sysdeps/mach/hurd/fcntl.c>
