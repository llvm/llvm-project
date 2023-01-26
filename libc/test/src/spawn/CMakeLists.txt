add_libc_testsuite(libc_spawn_unittests)

add_libc_unittest(
  posix_spawn_file_actions_test
  SUITE
    libc_spawn_unittests
  SRCS
    posix_spawn_file_actions_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.spawn
    libc.src.spawn.file_actions
    libc.src.spawn.posix_spawn_file_actions_addclose
    libc.src.spawn.posix_spawn_file_actions_adddup2
    libc.src.spawn.posix_spawn_file_actions_addopen
    libc.src.spawn.posix_spawn_file_actions_destroy
    libc.src.spawn.posix_spawn_file_actions_init
)
