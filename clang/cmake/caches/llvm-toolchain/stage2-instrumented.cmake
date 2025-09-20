# Stage 2 instrumented:
# * Build an instrumented clang, so we can generate profile data for stage 2.


set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(LLVM_BUILD_INSTRUMENTED IR CACHE STRING "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS -C ${CMAKE_CURRENT_LIST_DIR}/stage2.cmake CACHE STRING "")
set(CLANG_BOOTSTRAP_TARGETS clang check-all distribution install-distribution clang-bolt CACHE STRING "")
set(CLANG_BOLT OFF CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/stage2.cmake)
