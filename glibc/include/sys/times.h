#ifndef _SYS_TIMES_H
#include <posix/sys/times.h>

#ifndef _ISOMAC
/* Now define the internal interfaces.  */
extern clock_t __times (struct tms *__buffer);
#endif
#endif
