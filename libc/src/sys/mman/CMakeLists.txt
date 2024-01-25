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
