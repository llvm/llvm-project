if(NOT (TARGET libc.src.__support.threads.mutex))
  # Not all platforms have a mutex implementation. If mutex is unvailable,
  # we just skip everything about files.
  return()
endif()

add_object_library(
  file
  SRCS
    file.cpp
  HDRS
    file.h
  DEPENDS
    libc.src.__support.threads.mutex
    libc.include.errno
    libc.src.errno.errno
)
