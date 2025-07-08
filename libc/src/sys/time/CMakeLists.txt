if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  utimes
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.utimes
)

add_entrypoint_object(
  setitimer
  ALIAS
  DEPENDS
  .${LIBC_TARGET_OS}.setitimer 
)

add_entrypoint_object(
  getitimer
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.getitimer 
)
