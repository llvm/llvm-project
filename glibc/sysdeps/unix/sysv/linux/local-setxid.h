/* SETxID functions which only have to change the local thread and
   none of the possible other threads.  */
#include <sysdep.h>

#ifdef __NR_setresuid32
# define local_seteuid(id) INLINE_SYSCALL (setresuid32, 3, -1, id, -1)
#else
# define local_seteuid(id) INLINE_SYSCALL (setresuid, 3, -1, id, -1)
#endif


#ifdef __NR_setresgid32
# define local_setegid(id) INLINE_SYSCALL (setresgid32, 3, -1, id, -1)
#else
# define local_setegid(id) INLINE_SYSCALL (setresgid, 3, -1, id, -1)
#endif
