if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  eventfd
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.eventfd
)

add_entrypoint_object(
  eventfd_read
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.eventfd_read
)

add_entrypoint_object(
  eventfd_write
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.eventfd_write
)
