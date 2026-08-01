if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  if_indextoname
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.if_indextoname
)

add_entrypoint_object(
  if_nametoindex
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.if_nametoindex
)
