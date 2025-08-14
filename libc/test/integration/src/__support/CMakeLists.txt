add_subdirectory(threads)
if(LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(GPU)
endif()

add_libc_integration_test_suite(libc-support-integration-tests)

if(LIBC_TARGET_OS_IS_LINUX)
  add_integration_test(
    mpmc_stack_test
    SUITE
      libc-support-integration-tests
    SRCS
      mpmc_stack_test.cpp
    DEPENDS
      libc.src.__support.mpmc_stack
      libc.src.__support.threads.thread
      libc.src.pthread.pthread_create
      libc.src.pthread.pthread_join
  )
endif()
