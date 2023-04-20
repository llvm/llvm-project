add_executable(nvptx_loader Loader.cpp)
add_dependencies(nvptx_loader libc.src.__support.RPC.rpc)

target_link_libraries(nvptx_loader
  PRIVATE
  gpu_loader
  CUDA::cuda_driver
)
