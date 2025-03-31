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

if (LIBC_CONF_ENABLE_MALLOC_THREAD_CLEANUP)
  add_integration_test(
    malloc_cleanup_test
    SUITE
      libc-support-threads-integration-tests
    SRCS
      malloc_cleanup_test.cpp
    DEPENDS
      libc.src.__support.threads.thread
      libc.src.stdlib.atexit
      libc.src.stdlib.exit
      libc.src.stdlib._Exit
  )
endif()

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
