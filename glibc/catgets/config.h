#ifndef _CG_CONFIG_H
#define _CG_CONFIG_H

/* Use the internal textdomain used for libc messages.  */
#define PACKAGE _libc_intl_domainname
#ifndef VERSION
/* Get libc version number.  */
#include "../version.h"
#endif


#include_next <config.h>

#endif
