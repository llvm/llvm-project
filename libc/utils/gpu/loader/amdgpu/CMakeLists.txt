add_executable(amdhsa_loader Loader.cpp)
add_dependencies(amdhsa_loader libc.src.__support.RPC.rpc)

target_link_libraries(amdhsa_loader
  PRIVATE
  hsa-runtime64::hsa-runtime64
  gpu_loader
  llvmlibc_rpc_server
)
