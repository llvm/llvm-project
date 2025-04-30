#ifndef _SPAWN_H
#include <posix/spawn.h>

# ifndef _ISOMAC
__typeof (posix_spawn) __posix_spawn;
libc_hidden_proto (__posix_spawn)

__typeof (posix_spawn_file_actions_addclose)
  __posix_spawn_file_actions_addclose attribute_hidden;

__typeof (posix_spawn_file_actions_adddup2)
  __posix_spawn_file_actions_adddup2 attribute_hidden;

__typeof (posix_spawn_file_actions_addopen)
  __posix_spawn_file_actions_addopen attribute_hidden;

__typeof (posix_spawn_file_actions_destroy)
  __posix_spawn_file_actions_destroy attribute_hidden;

__typeof (posix_spawn_file_actions_init) __posix_spawn_file_actions_init
  attribute_hidden;

__typeof (posix_spawnattr_init) __posix_spawnattr_init
  attribute_hidden;

__typeof (posix_spawnattr_destroy) __posix_spawnattr_destroy
  attribute_hidden;

__typeof (posix_spawnattr_setflags) __posix_spawnattr_setflags
  attribute_hidden;

__typeof (posix_spawnattr_setsigdefault) __posix_spawnattr_setsigdefault
  attribute_hidden;

__typeof (posix_spawnattr_setsigmask) __posix_spawnattr_setsigmask
  attribute_hidden;

# endif /* !_ISOMAC  */
#endif /* spawn.h  */
