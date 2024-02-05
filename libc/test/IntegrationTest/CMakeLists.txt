if(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
  set(TEST_COMPILE_FLAGS
    -mcpu=${LIBC_GPU_TARGET_ARCHITECTURE}
    -emit-llvm # AMDGPU's intermediate object file format is bitcode.
    --target=${LIBC_GPU_TARGET_TRIPLE}
    -mcode-object-version=${LIBC_GPU_CODE_OBJECT_VERSION} # Manually set the ABI.
  )
elseif(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  set(TEST_COMPILE_FLAGS
    -march=${LIBC_GPU_TARGET_ARCHITECTURE}
    --target=${LIBC_GPU_TARGET_TRIPLE}
    --cuda-path=${LIBC_CUDA_ROOT}
  )
endif()

add_object_library(
  test
  NO_GPU_BUNDLE # Compile this file directly without special GPU handling.
  SRCS
    test.cpp
  COMPILE_OPTIONS
    ${TEST_COMPILE_FLAGS}
  HDRS
    test.h
  DEPENDS
    libc.src.__support.OSUtil.osutil
)
