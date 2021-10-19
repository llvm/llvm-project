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

add_subdirectory(config)
add_subdirectory(integration)
add_subdirectory(loader)
