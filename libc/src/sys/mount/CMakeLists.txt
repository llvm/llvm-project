if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  mount
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mount
)

add_entrypoint_object(
  umount
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.umount
)

add_entrypoint_object(
  umount2
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.umount2
)
