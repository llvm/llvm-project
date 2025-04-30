#ifndef _ULIMIT_H
#include <resource/ulimit.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern long int __ulimit (int __cmd, ...);
#endif
#endif
