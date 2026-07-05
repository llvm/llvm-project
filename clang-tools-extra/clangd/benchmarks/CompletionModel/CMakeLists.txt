add_benchmark(DecisionForestBenchmark DecisionForestBenchmark.cpp)

target_link_libraries(DecisionForestBenchmark
  PRIVATE
  clangDaemon
  LLVMSupport
  )
