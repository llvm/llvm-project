/* Add the include here so that we can redefine __fxprintf.  */
#include <stdio.h>

/* Rename symbols for protected names used in libc itself.  */
#define __ioctl ioctl
#define __socket socket
#define __strchrnul strchrnul
#define __strncasecmp strncasecmp

#define __fxprintf(args...) /* ignore */


#include "../resolv/res_hconf.c"
