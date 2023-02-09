add_subdirectory(UnitTest)

if(LLVM_LIBC_FULL_BUILD AND NOT
  (LIBC_TARGET_ARCHITECTURE_IS_GPU OR LIBC_TARGET_OS_IS_BAREMETAL))
  add_subdirectory(IntegrationTest)
endif()

add_header_library(
  errno_setter_matcher
  HDRS
    ErrnoSetterMatcher.h
)

add_custom_target(check-libc)
add_custom_target(check-llvmlibc)
add_dependencies(check-libc check-llvmlibc)

add_custom_target(exhaustive-check-libc)
add_custom_target(libc-long-running-tests)

add_subdirectory(src)
add_subdirectory(utils)

if(NOT LLVM_LIBC_FULL_BUILD)
  return()
endif()

if(NOT (${LIBC_TARGET_OS} STREQUAL "linux"))
  # Integration tests are currently only available for linux.
  return()
endif()
add_subdirectory(integration)
