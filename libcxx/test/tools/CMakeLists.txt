
set(LIBCXX_TEST_TOOLS_PATH ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Clang-tidy tests are disabled due to non-clang based compiler.")
  return()
endif()
add_subdirectory(clang_tidy_checks)
