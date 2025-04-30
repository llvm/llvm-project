#include <misc/sys/auxv.h>

#ifndef _ISOMAC

extern __typeof (getauxval) __getauxval;
libc_hidden_proto (__getauxval)

/* Like getauxval, but writes the value to *RESULT and returns true if
   found, or returns false.  Does not set errno.  */
_Bool __getauxval2 (unsigned long int type, unsigned long int *result);
libc_hidden_proto (__getauxval2)

#endif  /* !_ISOMAC */
