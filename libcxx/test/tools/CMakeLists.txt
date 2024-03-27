
set(LIBCXX_TEST_TOOLS_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

# TODO: Remove LIBCXX_ENABLE_CLANG_TIDY
if(LIBCXX_ENABLE_CLANG_TIDY)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Clang-tidy can only be used when building libc++ with "
                   "a clang compiler.")
    return()
  endif()
  add_subdirectory(clang_tidy_checks)
endif()
