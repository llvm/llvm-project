if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  raise
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.raise
)

add_entrypoint_object(
  kill
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.kill
)

add_entrypoint_object(
  sigaction
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigaction
)

add_entrypoint_object(
  sigaltstack
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigaltstack
)

add_entrypoint_object(
  sigprocmask
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigprocmask
)

add_entrypoint_object(
  sigemptyset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigemptyset
)

add_entrypoint_object(
  sigaddset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigaddset
)

add_entrypoint_object(
  signal
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.signal
)

add_entrypoint_object(
  sigfillset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigfillset
)

add_entrypoint_object(
  sigdelset
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sigdelset
)
