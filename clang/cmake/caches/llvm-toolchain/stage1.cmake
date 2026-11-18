# Stage 1
# * Build an LTO optimized libcxx, so we can staticially link it into stage 2
#   clang.


set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES compiler-rt libcxx libcxxabi libunwind CACHE STRING "")
set(LLVM_ENABLE_PROJECTS clang lld CACHE STRING "")

set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS -C ${CMAKE_CURRENT_LIST_DIR}/stage2-instrumented.cmake CACHE BOOL "")
set(BOOTSTRAP_LLVM_BUILD_INSTRUMENTED IR CACHE BOOL "")
set(CLANG_BOOTSTRAP_TARGETS stage2-check-all stage2-distribution stage2-install-distribution stage2-clang stage2-clang-bolt CACHE BOOL "")
set(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE STRING "")
set(RUNTIMES_CMAKE_ARGS "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON" CACHE STRING "")
set(LLVM_ENABLE_LLD ON CACHE STRING "")
