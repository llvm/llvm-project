if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_entrypoint_object(
  call_once
  SRCS
    call_once.cpp
  HDRS
    call_once.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.callonce
)

add_entrypoint_object(
  thrd_create
  SRCS
    thrd_create.cpp
  HDRS
    thrd_create.h
  DEPENDS
    libc.src.__support.threads.thread
    libc.include.threads
    libc.src.errno.errno
  COMPILE_OPTIONS
    ${libc_opt_high_flag}
    -fno-omit-frame-pointer # This allows us to sniff out the thread args from
                            # the new thread's stack reliably.
)

add_entrypoint_object(
  thrd_join
  SRCS
    thrd_join.cpp
  HDRS
    thrd_join.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  thrd_detach
  SRCS
    thrd_detach.cpp
  HDRS
    thrd_detach.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  thrd_current
  SRCS
    thrd_current.cpp
  HDRS
    thrd_current.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  thrd_equal
  SRCS
    thrd_equal.cpp
  HDRS
    thrd_equal.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  thrd_exit
  SRCS
    thrd_exit.cpp
  HDRS
    thrd_exit.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.thread
)

add_entrypoint_object(
  mtx_init
  SRCS
    mtx_init.cpp
  HDRS
    mtx_init.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  mtx_destroy
  SRCS
    mtx_destroy.cpp
  HDRS
    mtx_destroy.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  mtx_lock
  SRCS
    mtx_lock.cpp
  HDRS
    mtx_lock.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  mtx_unlock
  SRCS
    mtx_unlock.cpp
  HDRS
    mtx_unlock.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  tss_create
  SRCS
    tss_create.cpp
  HDRS
    tss_create.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  tss_delete
  SRCS
    tss_delete.cpp
  HDRS
    tss_delete.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  tss_get
  SRCS
    tss_get.cpp
  HDRS
    tss_get.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
)

add_entrypoint_object(
  tss_set
  SRCS
    tss_set.cpp
  HDRS
    tss_set.h
  DEPENDS
    libc.include.threads
    libc.src.__support.threads.mutex
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
