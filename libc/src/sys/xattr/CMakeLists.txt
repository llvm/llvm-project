if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  fgetxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fgetxattr
)

add_entrypoint_object(
  getxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getxattr
)

add_entrypoint_object(
  lgetxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.lgetxattr
)

add_entrypoint_object(
  fsetxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fsetxattr
)

add_entrypoint_object(
  setxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.setxattr
)

add_entrypoint_object(
  lsetxattr
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.lsetxattr
)

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
