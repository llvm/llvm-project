if(NOT (TARGET libc.src.__support.threads.thread AND
        TARGET libc.src.__support.threads.mutex))
  return()
endif()

add_libc_integration_test_suite(libc-support-threads-integration-tests)

add_integration_test(
  thread_detach_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    thread_detach_test.cpp
  LOADER
    libc.loader.linux.crt1
  DEPENDS
    libc.src.__support.threads.mutex
    libc.src.__support.threads.thread
)
