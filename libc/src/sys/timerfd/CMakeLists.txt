if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  timerfd_create
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.timerfd_create
)

add_entrypoint_object(
  timerfd_gettime
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.timerfd_gettime
)

add_entrypoint_object(
  timerfd_settime
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.timerfd_settime
)
