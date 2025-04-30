/* Although Alpha defines _DIRENT_MATCHES_DIRENT64, 'struct dirent' and
   'struct dirent64' have slight different internal layout with d_ino
   being a __ino_t on non-LFS version with an extra __pad field which should
   be zeroed.  */

#include <dirent.h>
/* It suppresses the __getdents64 to __getdents alias.  */
#undef _DIRENT_MATCHES_DIRENT64
#define _DIRENT_MATCHES_DIRENT64 0
#include <sysdeps/unix/sysv/linux/getdents64.c>
