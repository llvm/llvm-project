set(LLVM_LINK_COMPONENTS
  Support
  )

add_benchmark(ClangCallDescriptionMapBenchmark
  CallDescriptionMapBenchmark.cpp
  )

clang_target_link_libraries(ClangCallDescriptionMapBenchmark
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangStaticAnalyzerCore
  clangTooling
  )
