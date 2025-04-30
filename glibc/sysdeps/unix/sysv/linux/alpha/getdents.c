/* Although Alpha defines _DIRENT_MATCHES_DIRENT64, 'struct dirent' and
   'struct dirent64' have slight different internal layout with d_ino
   being a __ino_t on non-LFS version with an extra __pad field which should
   be zeroed.  */

#include <dirent.h>
#undef _DIRENT_MATCHES_DIRENT64
#define _DIRENT_MATCHES_DIRENT64 0
#define DIRENT_SET_DP_INO(dp, value) \
  do { (dp)->d_ino = (value); (dp)->__pad = 0; } while (0)
#include <sysdeps/unix/sysv/linux/getdents.c>
