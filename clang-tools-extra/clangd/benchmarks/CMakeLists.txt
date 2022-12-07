if(CLANGD_DECISION_FOREST)
  add_subdirectory(CompletionModel)
endif()

add_benchmark(IndexBenchmark IndexBenchmark.cpp)

target_link_libraries(IndexBenchmark
  PRIVATE
  clangDaemon
  LLVMSupport
  )
