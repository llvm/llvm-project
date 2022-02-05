if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

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
