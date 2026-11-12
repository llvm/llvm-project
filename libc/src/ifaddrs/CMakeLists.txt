if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  freeifaddrs
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.freeifaddrs
)

add_entrypoint_object(
  getifaddrs
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getifaddrs
)
