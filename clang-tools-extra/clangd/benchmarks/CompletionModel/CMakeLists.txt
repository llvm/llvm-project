set(LLVM_LINK_COMPONENTS
  Support
  )

add_benchmark(DecisionForestBenchmark DecisionForestBenchmark.cpp)

target_link_libraries(DecisionForestBenchmark
  PRIVATE
  clangDaemon
  )
