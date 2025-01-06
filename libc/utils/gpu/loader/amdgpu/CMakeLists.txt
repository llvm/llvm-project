set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Object
  Option
  Support
  FrontendOffloading
  )

add_llvm_executable(amdhsa-loader amdhsa-loader.cpp)

target_link_libraries(amdhsa-loader
  PRIVATE
  hsa-runtime64::hsa-runtime64
  gpu_loader
  llvmlibc_rpc_server
)
