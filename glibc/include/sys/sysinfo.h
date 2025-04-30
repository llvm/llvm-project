#ifndef _SYS_SYSINFO_H
#include_next <sys/sysinfo.h>

# ifndef _ISOMAC

/* Now we define the internal interface.  */

/* Return number of configured processors.  */
extern int __get_nprocs_conf (void);
libc_hidden_proto (__get_nprocs_conf)

/* Return number of available processors.  */
extern int __get_nprocs (void);
libc_hidden_proto (__get_nprocs)

/* Return number of physical pages of memory in the system.  */
extern long int __get_phys_pages (void);
libc_hidden_proto (__get_phys_pages)

/* Return number of available physical pages of memory in the system.  */
extern long int __get_avphys_pages (void);
libc_hidden_proto (__get_avphys_pages)

/* Return maximum number of processes this real user ID can have.  */
extern long int __get_child_max (void) attribute_hidden;

# endif /* !_ISOMAC */
#endif /* sys/sysinfo.h */
