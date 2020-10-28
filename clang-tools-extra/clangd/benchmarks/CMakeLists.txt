include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(CompletionModel)

add_benchmark(IndexBenchmark IndexBenchmark.cpp)

target_link_libraries(IndexBenchmark
  PRIVATE
  clangDaemon
  LLVMSupport
  )
