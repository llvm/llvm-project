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
  )
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
