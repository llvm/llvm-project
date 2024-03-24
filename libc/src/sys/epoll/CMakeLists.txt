if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  epoll_wait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.epoll_wait
)

add_entrypoint_object(
  epoll_pwait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.epoll_pwait
)

add_entrypoint_object(
  epoll_pwait2
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.epoll_pwait2
)
