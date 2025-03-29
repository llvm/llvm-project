if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_header_library(
  file_actions
  HDRS
    file_actions.h
  DEPENDS
    libc.include.spawn
)

add_entrypoint_object(
  posix_spawn_file_actions_init
  SRCS
    posix_spawn_file_actions_init.cpp
  HDRS
    posix_spawn_file_actions_init.h
  DEPENDS
    libc.include.spawn
)

add_entrypoint_object(
  posix_spawn_file_actions_destroy
  SRCS
    posix_spawn_file_actions_destroy.cpp
  HDRS
    posix_spawn_file_actions_destroy.h
  DEPENDS
    .file_actions
    libc.include.spawn
    libc.src.__support.CPP.new
    libc.src.errno.errno
)

add_entrypoint_object(
  posix_spawn_file_actions_adddup2
  SRCS
    posix_spawn_file_actions_adddup2.cpp
  HDRS
    posix_spawn_file_actions_adddup2.h
  DEPENDS
    .file_actions
    libc.include.spawn
    libc.src.__support.CPP.new
    libc.src.errno.errno
)

add_entrypoint_object(
  posix_spawn_file_actions_addopen
  SRCS
    posix_spawn_file_actions_addopen.cpp
  HDRS
    posix_spawn_file_actions_addopen.h
  DEPENDS
    .file_actions
    libc.include.spawn
    libc.src.__support.CPP.new
    libc.src.errno.errno
)

add_entrypoint_object(
  posix_spawn_file_actions_addclose
  SRCS
    posix_spawn_file_actions_addclose.cpp
  HDRS
    posix_spawn_file_actions_addclose.h
  DEPENDS
    .file_actions
    libc.include.spawn
    libc.src.__support.CPP.new
    libc.src.errno.errno
)

add_entrypoint_object(
  posix_spawn
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.posix_spawn
)
