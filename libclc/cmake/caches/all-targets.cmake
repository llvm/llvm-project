set(LLVM_ENABLE_PROJECTS "clang" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "AMDGPU;NVPTX" CACHE STRING "")

set(LLVM_RUNTIME_TARGETS
  amdgcn-amd-amdhsa-llvm
  nvptx64-nvidia-cuda
  spirv32-unknown-unknown
  spirv64-unknown-unknown
  spirv32-unknown-vulkan
  spirv64-unknown-vulkan
  CACHE STRING "")

set(RUNTIMES_amdgcn-amd-amdhsa-llvm_LLVM_ENABLE_RUNTIMES "libclc" CACHE STRING "")

set(RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES "libclc" CACHE STRING "")

set(RUNTIMES_spirv32-unknown-unknown_LLVM_ENABLE_RUNTIMES "libclc" CACHE STRING "")
set(RUNTIMES_spirv64-unknown-unknown_LLVM_ENABLE_RUNTIMES "libclc" CACHE STRING "")

set(RUNTIMES_spirv32-unknown-vulkan_LLVM_ENABLE_RUNTIMES "libclc" CACHE STRING "")
set(RUNTIMES_spirv64-unknown-vulkan_LLVM_ENABLE_RUNTIMES "libclc" CACHE STRING "")
