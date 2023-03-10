add_executable(amdhsa_loader Loader.cpp)
add_dependencies(amdhsa_loader libc.src.__support.RPC.rpc)

target_include_directories(amdhsa_loader PRIVATE ${LIBC_SOURCE_DIR})
target_link_libraries(amdhsa_loader
  PRIVATE
  hsa-runtime64::hsa-runtime64
  gpu_loader
)
