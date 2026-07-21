if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  semget
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.semget
)

add_entrypoint_object(
  semctl
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.semctl
)

add_entrypoint_object(
  semop
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.semop
)
