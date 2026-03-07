set(LLVM_LINK_COMPONENTS
  Support
  )

add_benchmark(ClangDocBenchmark ClangDocBenchmark.cpp)

target_include_directories(ClangDocBenchmark
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(ClangDocBenchmark
  PRIVATE
  clangDoc
  clangTooling
  clangBasic
  clangAST
  clangFrontend
  clangSerialization
)
