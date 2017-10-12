set(LLVM_LINK_COMPONENTS support)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")

add_clang_executable(clang-format-fuzzer
  EXCLUDE_FROM_ALL
  ClangFormatFuzzer.cpp
  )

target_link_libraries(clang-format-fuzzer
  ${CLANG_FORMAT_LIB_DEPS})
