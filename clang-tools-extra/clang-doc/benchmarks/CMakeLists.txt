set(LLVM_LINK_COMPONENTS
  Support
  )

get_filename_component(CLANG_DOC_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-doc REALPATH)
include_directories(
  ${CLANG_DOC_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )

add_benchmark(ClangDocBenchmark ClangDocBenchmark.cpp)

target_link_libraries(ClangDocBenchmark
  PRIVATE
  clangDoc
  clangTooling
  clangBasic
  clangAST
)