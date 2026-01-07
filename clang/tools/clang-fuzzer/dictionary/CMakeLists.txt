set(CMAKE_CXX_FLAGS ${CXX_FLAGS_NOFUZZ})
add_clang_executable(clang-fuzzer-dictionary
  dictionary.c

  DISABLE_PCH_REUSE # no C++ files, prevent CMake error.
  )

