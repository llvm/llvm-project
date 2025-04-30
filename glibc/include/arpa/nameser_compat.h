#ifndef _ARPA_NAMESER_COMPAT_
#include <resolv/arpa/nameser_compat.h>

# ifndef _ISOMAC

/* The number is outside the 16-bit RR type range and is used
   internally by the implementation.  */
#define T_QUERY_A_AND_AAAA 439963904

# endif /* !_ISOMAC */
#endif
