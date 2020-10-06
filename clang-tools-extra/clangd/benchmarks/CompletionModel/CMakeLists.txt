include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_benchmark(DecisionForestBenchmark DecisionForestBenchmark.cpp)

target_link_libraries(DecisionForestBenchmark
  PRIVATE
  clangDaemon
  LLVMSupport
  )
