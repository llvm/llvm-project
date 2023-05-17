if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  wait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.wait
)

add_entrypoint_object(
  wait4
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.wait4
)

add_entrypoint_object(
  waitpid
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.waitpid
)
