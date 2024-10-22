foreach(target nvptx amdgpu)
  add_subdirectory(${target})
  list(APPEND target_gpu_timing libc.benchmarks.gpu.timing.${target}.${target}_timing)
endforeach()

add_header_library(
  timing
  HDRS
    timing.h
  DEPENDS
    ${target_gpu_timing}
)
