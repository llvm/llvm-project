if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  return()
endif()

foreach(target nvptx amdgpu generic)
  add_subdirectory(${target})
  list(APPEND target_gpu_utils libc.src.__support.GPU.${target}.${target}_utils)
endforeach()

add_header_library(
  utils
  HDRS
    utils.h
  DEPENDS
    ${target_gpu_utils}
)
