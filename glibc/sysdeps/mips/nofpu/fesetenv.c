/* MIPS bits/fenv.h used to define exception macros for soft-float
   despite that not supporting exceptions.  Ensure use of the old
   FE_NOMASK_ENV value still produces errors (see bug 17088).  */
#include <fenv.h>
#undef FE_ALL_EXCEPT
#define FE_ALL_EXCEPT 0x7c
#define FE_NOMASK_ENV ((const fenv_t *) -2)
#include <math/fesetenv.c>
