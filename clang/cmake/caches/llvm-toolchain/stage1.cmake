# Stage 1:
# * Build the prerequisites for stage 2.
# * We will be building an LTO optimized libcxx in stage 2, so we need to
#   build clang and lld.


set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "")

set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS -C ${CMAKE_CURRENT_LIST_DIR}/stage2.cmake CACHE BOOL "")
set(CLANG_BOOTSTRAP_TARGETS stage3-check-all stage3-distribution stage3-install-distribution stage3-clang stage3-clang-bolt CACHE BOOL "")
