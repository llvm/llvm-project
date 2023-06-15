add_executable(nvptx_loader Loader.cpp)
add_dependencies(nvptx_loader libc.src.__support.RPC.rpc)

if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(nvptx_loader PRIVATE -fno-rtti)
endif()
target_include_directories(nvptx_loader PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(nvptx_loader
  PRIVATE
  gpu_loader
  llvmlibc_rpc_server
  CUDA::cuda_driver
  LLVMObject
  LLVMSupport
)
