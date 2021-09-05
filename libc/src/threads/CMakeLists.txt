if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  call_once
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.call_once
)

add_entrypoint_object(
  thrd_create
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.thrd_create
)

add_entrypoint_object(
  thrd_join
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.thrd_join
)

add_entrypoint_object(
  mtx_init
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mtx_init
)

add_entrypoint_object(
  mtx_destroy
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mtx_destroy
)

add_entrypoint_object(
  mtx_lock
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mtx_lock
)

add_entrypoint_object(
  mtx_unlock
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.mtx_unlock
)

add_entrypoint_object(
  cnd_init
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.cnd_init
)

add_entrypoint_object(
  cnd_destroy
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.cnd_destroy
)

add_entrypoint_object(
  cnd_wait
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.cnd_wait
)

add_entrypoint_object(
  cnd_signal
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.cnd_signal
)

add_entrypoint_object(
  cnd_broadcast
  ALIAS
  DEPENDS
    .${LIBC_TARGET_OS}.cnd_broadcast
)
