if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  socket
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.socket
)

add_entrypoint_object(
  socketpair
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.socketpair
)

add_entrypoint_object(
  send
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.send
)

add_entrypoint_object(
  sendto
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sendto
)

add_entrypoint_object(
  sendmsg
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.sendmsg
)

add_entrypoint_object(
  recv
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.recv
)

add_entrypoint_object(
  recvfrom
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.recvfrom
)

add_entrypoint_object(
  recvmsg
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.recvmsg
)
