if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  access
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.access
)

add_entrypoint_object(
  chdir
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.chdir
)

add_entrypoint_object(
  close
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.close
)

add_entrypoint_object(
  dup
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.dup
)

add_entrypoint_object(
  dup2
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.dup2
)

add_entrypoint_object(
  dup3
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.dup3
)

add_entrypoint_object(
  fchdir
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fchdir
)

add_entrypoint_object(
  fsync
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fsync
)

add_entrypoint_object(
  ftruncate
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.ftruncate
)

add_entrypoint_object(
  getpid
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getpid
)

add_entrypoint_object(
  getppid
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getppid
)

add_entrypoint_object(
  geteuid
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.geteuid
)

add_entrypoint_object(
  getuid
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getuid
)

add_entrypoint_object(
  link
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.link
)

add_entrypoint_object(
  linkat
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.linkat
)

add_entrypoint_object(
  lseek
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.lseek
)

add_entrypoint_object(
  pread
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.pread
)

add_entrypoint_object(
  pwrite
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.pwrite
)

add_entrypoint_object(
  read
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.read
)

add_entrypoint_object(
  readlink
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.readlink
)

add_entrypoint_object(
  readlinkat
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.readlinkat
)

add_entrypoint_object(
  rmdir
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.rmdir
)

add_entrypoint_object(
  symlink
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.symlink
)

add_entrypoint_object(
  symlinkat
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.symlinkat
)

add_entrypoint_object(
  truncate
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.truncate
)

add_entrypoint_object(
  unlink
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.unlink
)

add_entrypoint_object(
  unlinkat
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.unlinkat
)

add_entrypoint_object(
  write
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.write
)
