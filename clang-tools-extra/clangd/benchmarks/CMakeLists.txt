add_subdirectory(CompletionModel)

add_benchmark(IndexBenchmark IndexBenchmark.cpp)

target_link_libraries(IndexBenchmark
  PRIVATE
  clangDaemon
  LLVMSupport
  )
