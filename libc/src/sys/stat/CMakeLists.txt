if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  chmod
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.chmod
)

add_entrypoint_object(
  fchmodat
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fchmodat
)

add_entrypoint_object(
  fchmod
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.fchmod
)

add_entrypoint_object(
  mkdir
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mkdir
)

add_entrypoint_object(
  mkdirat
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mkdirat
)
