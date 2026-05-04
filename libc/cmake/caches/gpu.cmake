set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;lld" CACHE STRING "")
set(LLVM_RUNTIME_TARGETS default;amdgcn-amd-amdhsa;nvptx64-nvidia-cuda CACHE STRING "") 
set(RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES "compiler-rt;libc" CACHE STRING "")
set(RUNTIMES_amdgcn-amd-amdhsa_LLVM_ENABLE_RUNTIMES "compiler-rt;libc" CACHE STRING "")
