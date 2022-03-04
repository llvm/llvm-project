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
  SRCS
    mtx_init.cpp
  HDRS
    mtx_init.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  mtx_destroy
  SRCS
    mtx_destroy.cpp
  HDRS
    mtx_destroy.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  mtx_lock
  SRCS
    mtx_lock.cpp
  HDRS
    mtx_lock.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  mtx_unlock
  SRCS
    mtx_unlock.cpp
  HDRS
    mtx_unlock.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
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
