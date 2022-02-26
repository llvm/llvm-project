if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

add_header_library(
  thread
  HDRS
    mutex.h
  DEPS
    .${LIBC_TARGET_OS}.thread
)
