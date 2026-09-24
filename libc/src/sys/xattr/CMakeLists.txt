if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  flistxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.flistxattr
)

add_entrypoint_object(
  listxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.listxattr
)

add_entrypoint_object(
  llistxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.llistxattr
)
