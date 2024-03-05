if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  return()
endif()

add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
set(target_gpu_utils libc.src.__support.GPU.${LIBC_TARGET_ARCHITECTURE}.${LIBC_TARGET_ARCHITECTURE}_utils)

add_header_library(
  utils
  HDRS
    utils.h
  DEPENDS
    ${target_gpu_utils}
)
