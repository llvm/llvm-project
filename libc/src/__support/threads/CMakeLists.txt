add_header_library(
  mutex_common
  HDRS
    mutex_common.h
)

add_header_library(
  thread_attrib
  HDRS
    thread_attrib.h
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
  add_header_library(
    thread
    HDRS
      thread.h
    DEPENDS
      .${LIBC_TARGET_OS}.thread
  )
endif()
