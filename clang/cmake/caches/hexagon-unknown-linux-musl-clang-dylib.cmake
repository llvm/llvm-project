set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "")
set(CLANG_LINK_LLVM_DYLIB ON CACHE BOOL "")

# Clear version suffix to prevent versioned library names like libLLVM.so.22.1-rc1
# which lld doesn't recognize. This results in libLLVM.so.22.1 instead.
set(LLVM_VERSION_SUFFIX "" CACHE STRING "")
