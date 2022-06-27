add_header_library(
  mutex_common
  HDRS
    mutex_common.h
)

add_header_library(
  thread_attrib
  HDRS
    thread_attrib.h
  DEPENDS
    libc.src.__support.common
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
endif()

if(TARGET libc.src.__support.threads.${LIBC_TARGET_OS}.thread)
  add_object_library(
    thread
    HDRS
      thread.h
    SRCS
      ${LIBC_TARGET_OS}/thread.cpp
    DEPENDS
      .thread_attrib
    COMPILE_OPTIONS
      -O3
      -fno-omit-frame-pointer # This allows us to sniff out the thread args from
                              # the new thread's stack reliably.
  )
endif()
