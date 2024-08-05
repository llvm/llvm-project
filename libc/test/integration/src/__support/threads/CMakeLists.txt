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
  DEPENDS
    libc.src.__support.threads.mutex
    libc.src.__support.threads.thread
)

add_integration_test(
  thread_tls_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    thread_tls_test.cpp
  DEPENDS
    libc.src.__support.threads.thread
)

add_integration_test(
  lambda_lock_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    lambda_lock_test.cpp
  DEPENDS
    libc.src.__support.threads.lambda_lock
    libc.src.pthread.pthread_create
    libc.src.pthread.pthread_join
    libc.src.__support.CPP.stringstream
    libc.src.__support.CPP.limits
)
