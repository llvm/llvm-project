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
  main_exit_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    main_exit_test.cpp
  DEPENDS
    libc.src.__support.threads.thread
)

add_integration_test(
  double_exit_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    double_exit_test.cpp
  DEPENDS
    libc.src.__support.threads.thread
    libc.src.stdlib.exit
)

add_integration_test(
  futex_requeue_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    futex_requeue_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.include.pthread
    libc.src.pthread.pthread_create
    libc.src.pthread.pthread_join
)

add_integration_test(
  cndvar_test
  SUITE
    libc-support-threads-integration-tests
  SRCS
    cndvar_test.cpp
  DEPENDS
    libc.hdr.time_macros
    libc.src.__support.CPP.expected
    libc.src.__support.threads.CndVar
    libc.src.__support.threads.mutex
    libc.src.__support.threads.mutex_common
    libc.src.__support.threads.sleep
    libc.src.__support.time.clock_gettime
    libc.src.threads.thrd_create
    libc.src.threads.thrd_join
)
