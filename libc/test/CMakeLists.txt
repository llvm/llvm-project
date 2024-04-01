add_custom_target(check-libc)
add_custom_target(libc-unit-tests)
add_custom_target(libc-hermetic-tests)
add_dependencies(check-libc libc-unit-tests libc-hermetic-tests)

add_custom_target(exhaustive-check-libc)
add_custom_target(libc-long-running-tests)

add_subdirectory(UnitTest)

if(LIBC_TARGET_OS_IS_GPU AND
   (NOT TARGET libc.utils.gpu.loader OR LIBC_GPU_TESTS_DISABLED))
  message(WARNING "Cannot build libc GPU tests, missing loader or architecture")
  return()
endif()

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(utils)

if(LLVM_LIBC_FULL_BUILD AND NOT LIBC_TARGET_OS_IS_BAREMETAL)
  add_subdirectory(IntegrationTest)
endif()

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

if(NOT ${LIBC_TARGET_OS} STREQUAL "linux" AND
   NOT ${LIBC_TARGET_OS} STREQUAL "gpu")
  # Integration tests are currently only available for linux and the GPU.
  return()
endif()
add_subdirectory(integration)
