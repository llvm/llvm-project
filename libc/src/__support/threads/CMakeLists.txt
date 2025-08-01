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
elseif(NOT (LIBC_CONF_THREAD_MODE STREQUAL LIBC_THREAD_MODE_PLATFORM))
  add_header_library(
    mutex
  HDRS
    mutex.h
  DEPENDS
    .mutex_common
  )
endif()

add_header_library(
  thread_common
  HDRS
    thread.h
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.common
    libc.src.__support.CPP.atomic
    libc.src.__support.CPP.optional
    libc.src.__support.CPP.string_view
    libc.src.__support.CPP.stringstream
)

if(TARGET libc.src.__support.threads.${LIBC_TARGET_OS}.thread)
  add_object_library(
    thread
    SRCS
      thread.cpp
    DEPENDS
      .mutex
      .${LIBC_TARGET_OS}.thread
      libc.hdr.stdint_proxy
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

if (LLVM_LIBC_FULL_BUILD)
  set(identifier_dependency_on_thread libc.src.__support.threads.thread)
endif()

add_header_library(
  identifier
  HDRS
    identifier.h
  DEPENDS
    libc.src.__support.OSUtil.osutil
    libc.src.__support.common
    libc.include.sys_syscall
    libc.hdr.types.pid_t
    ${identifier_dependency_on_thread}
)
