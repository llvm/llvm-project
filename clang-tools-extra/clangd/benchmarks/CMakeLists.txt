if(CLANGD_DECISION_FOREST)
  add_subdirectory(CompletionModel)
endif()

set(LLVM_LINK_COMPONENTS
  Support
  )

add_benchmark(IndexBenchmark IndexBenchmark.cpp)

target_link_libraries(IndexBenchmark
  PRIVATE
  clangDaemon
  )
