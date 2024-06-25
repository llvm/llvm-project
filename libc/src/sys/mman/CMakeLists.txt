if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  madvise
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.madvise
)

add_entrypoint_object(
  mmap
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mmap
)

add_entrypoint_object(
  munmap
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.munmap
)

add_entrypoint_object(
  mprotect
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mprotect
)

add_entrypoint_object(
  posix_madvise
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.posix_madvise
)

add_entrypoint_object(
  mincore
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mincore
)

add_entrypoint_object(
  mlock
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mlock
)

add_entrypoint_object(
  mlock2
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mlock2
)

add_entrypoint_object(
  munlock
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.munlock
)

add_entrypoint_object(
  mlockall
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mlockall
)

add_entrypoint_object(
  munlockall
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.munlockall
)

add_entrypoint_object(
  msync
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.msync
)

add_entrypoint_object(
  shm_open
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.shm_open
)

add_entrypoint_object(
  shm_unlink
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.shm_unlink
)
