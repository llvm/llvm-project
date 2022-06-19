add_benchmark(ClangPseudoBenchmark Benchmark.cpp)

target_link_libraries(ClangPseudoBenchmark
  PRIVATE
  clangPseudo
  clangPseudoGrammar
  LLVMSupport
  )
