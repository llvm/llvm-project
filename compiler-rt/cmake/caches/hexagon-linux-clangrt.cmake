
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR OFF CACHE BOOL "")
set(LLVM_DEFAULT_TARGET_TRIPLE hexagon-unknown-linux-musl CACHE STRING "")

set(COMPILER_RT_USE_LLVM_UNWINDER ON CACHE BOOL "")

# Some build failures here, including the inline asm in
# `compiler-rt/lib/sanitizer_common/sanitizer_redefine_builtins.h`, so
# we can just disable these for now:
set(COMPILER_RT_BUILD_BUILTINS OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_SANITIZERS OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_XRAY OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_MEMPROF OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_CTX_PROFILE OFF CACHE BOOL "")

