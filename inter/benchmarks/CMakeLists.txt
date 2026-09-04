find_path(INTER_LEVEL_ZERO_INCLUDE_DIR level_zero/ze_api.h REQUIRED)
find_library(INTER_LEVEL_ZERO_LIBRARY ze_loader REQUIRED)

add_executable(inter-lighthouse-benchmark lighthouse-benchmark.cpp)
target_include_directories(inter-lighthouse-benchmark PRIVATE
  ${INTER_LEVEL_ZERO_INCLUDE_DIR})
target_link_libraries(inter-lighthouse-benchmark PRIVATE
  ${INTER_LEVEL_ZERO_LIBRARY})
