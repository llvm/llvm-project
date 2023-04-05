add_startup_object(
  crt1
  SRC
    start.cpp
  DEPENDS
    libc.src.__support.RPC.rpc_client
  COMPILE_OPTIONS
    -ffreestanding # To avoid compiler warnings about calling the main function.
    -fno-builtin
    -nogpulib # Do not include any GPU vendor libraries.
    -march=${LIBC_GPU_TARGET_ARCHITECTURE}
    --target=${LIBC_GPU_TARGET_TRIPLE}
    --cuda-path=${LIBC_CUDA_ROOT}
  NO_GPU_BUNDLE # Compile this file directly without special GPU handling.
)
get_fq_target_name(crt1 fq_name)

# Ensure that clang uses the correct linker for this object type.
target_link_libraries(${fq_name}
  PUBLIC
  "-march=${LIBC_GPU_TARGET_ARCHITECTURE}"
  "--target=${LIBC_GPU_TARGET_TRIPLE}"
  "--cuda-path=${LIBC_CUDA_ROOT}"
)
