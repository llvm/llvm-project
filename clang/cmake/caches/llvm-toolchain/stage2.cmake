# Stage 2:
# * Build an LTO optimized libcxx, so we can staticially link it into stage 3
#   clang.
# * Stage 3 will be PGO optimized, so we need to build clang, lld, and
#   compiler-rt in stage 2.


set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;libcxx;libcxxabi;libunwind" CACHE STRING "" FORCE)
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "" FORCE)

set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS -C ${CMAKE_CURRENT_LIST_DIR}/stage3-instrumented.cmake CACHE BOOL "")
set(BOOTSTRAP_LLVM_BUILD_INSTRUMENTED IR CACHE BOOL "")
set(CLANG_BOOTSTRAP_TARGETS stage3-check-all stage3-distribution stage3-install-distribution stage3-clang stage3-clang-bolt CACHE BOOL "")
set(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY ON CACHE STRING "")
set(RUNTIMES_CMAKE_ARGS "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON" CACHE STRING "")
set(LLVM_ENABLE_LLD ON CACHE STRING "")
#set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
