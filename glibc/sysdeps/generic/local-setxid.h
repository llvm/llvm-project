/* No special support.  Fall back to the regular functions.  */

#define local_seteuid(id) seteuid (id)
#define local_setegid(id) setegid (id)
