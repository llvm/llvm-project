add_subdirectory(threads)
if(LIBC_TARGET_OS_IS_GPU)
  add_subdirectory(GPU)
endif()

add_libc_integration_test_suite(libc-support-integration-tests)
