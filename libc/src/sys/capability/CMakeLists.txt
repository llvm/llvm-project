if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  capget
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.capget
)

add_entrypoint_object(
  capset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.capset
)
