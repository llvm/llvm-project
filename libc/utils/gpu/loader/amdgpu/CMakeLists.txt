add_executable(amdhsa-loader Loader.cpp)

target_link_libraries(amdhsa-loader
  PRIVATE
  hsa-runtime64::hsa-runtime64
  gpu_loader
  llvmlibc_rpc_server
)
