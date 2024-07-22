add_header_library(
  mutex_common
  HDRS
    mutex_common.h
)

add_header_library(
  sleep
  HDRS
    sleep.h
)

add_header_library(
  spin_lock
  HDRS
    spin_lock.h
  DEPENDS
    .sleep
    libc.src.__support.CPP.atomic
)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

if(TARGET libc.src.__support.threads.${LIBC_TARGET_OS}.mutex)
  add_header_library(
    mutex
    HDRS
      mutex.h
    DEPENDS
      .${LIBC_TARGET_OS}.mutex
  )

  add_object_library(
    fork_callbacks
    SRCS
      fork_callbacks.cpp
    HDRS
      fork_callbacks.h
    DEPENDS
      .mutex
      libc.src.__support.CPP.mutex
  )
endif()

if (LIBC_CONF_ENABLE_TID_CACHE)
  set(libc_copt_enable_tid_cache 1)
else()
  set(libc_copt_enable_tid_cache 0)
endif()

add_header_library(
  thread_common
  HDRS
    thread.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.stringstream
    libc.hdr.types.pid_t
  COMPILE_OPTIONS
    -DLIBC_COPT_ENABLE_TID_CACHE=${libc_copt_enable_tid_cache}
)

if(TARGET libc.src.__support.threads.${LIBC_TARGET_OS}.thread)
  add_object_library(
    thread
    SRCS
      thread.cpp
    DEPENDS
      .mutex
      .${LIBC_TARGET_OS}.thread
      libc.src.__support.common
      libc.src.__support.fixedvector
      libc.src.__support.CPP.array
      libc.src.__support.CPP.mutex
      libc.src.__support.CPP.optional
  )
endif()

if(TARGET libc.src.__support.threads.${LIBC_TARGET_OS}.callonce)
  add_object_library(
    callonce
    ALIAS
    DEPENDS
      .${LIBC_TARGET_OS}.callonce
  )
endif()

if(TARGET libc.src.__support.threads.${LIBC_TARGET_OS}.CndVar)
  add_object_library(
    CndVar
    ALIAS
    DEPENDS
    .${LIBC_TARGET_OS}.CndVar
  )
endif()

set(tid_dep)
if (LLVM_LIBC_FULL_BUILD)
  list(APPEND tid_dep libc.src.__support.thread)
else()
  list(APPEND tid_dep libc.src.__support.OSUtil.osutil)
  list(APPEND tid_dep libc.include.sys_syscall)
endif()

add_header_library(
  tid
  HDRS
    tid.h
  DEPENDS
    libc.src.__support.common
    libc.hdr.types.pid_t
    ${tid_dep}
)
