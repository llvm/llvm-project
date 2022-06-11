if(NOT (TARGET libc.src.__support.threads.thread AND
        TARGET libc.src.__support.threads.mutex))
  return()
endif()

add_libc_unittest(
  thread_detach_test
  SUITE
  SRCS
    thread_detach_test.cpp
  DEPENDS
    libc.src.__support.threads.mutex
    libc.src.__support.threads.thread
    libc.src.__support.threads.thread_attrib
  COMPILE_OPTIONS
    -O3
    -fno-omit-frame-pointer
)
