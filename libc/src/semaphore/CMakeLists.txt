if(NOT TARGET libc.src.__support.OSUtil.osutil)
  return()
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  sem_init
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_init
)

add_entrypoint_object(
  sem_destroy
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_destroy
)

add_entrypoint_object(
  sem_getvalue
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_getvalue
)

add_entrypoint_object(
  sem_open
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_open
)

add_entrypoint_object(
  sem_close
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_close
)

add_entrypoint_object(
  sem_unlink
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_unlink
)

add_entrypoint_object(
  sem_post
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_post
)

add_entrypoint_object(
  sem_wait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_wait
)

add_entrypoint_object(
  sem_trywait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_trywait
)

add_entrypoint_object(
  sem_timedwait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_timedwait
)

add_entrypoint_object(
  sem_clockwait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sem_clockwait
)
