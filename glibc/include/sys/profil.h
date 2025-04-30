#ifndef _PROFIL_H
#include <gmon/sys/profil.h>

# ifndef _ISOMAC

/* Now define the internal interfaces.  */

extern int __sprofil (struct prof *__profp, int __profcnt,
		      struct timeval *__tvp, unsigned int __flags);

# endif /* !_ISOMAC */
#endif /* _PROFIL_H */
