if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  makedev
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.makedev
)

add_entrypoint_object(
  major
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.major
)

add_entrypoint_object(
  minor
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.minor
)
