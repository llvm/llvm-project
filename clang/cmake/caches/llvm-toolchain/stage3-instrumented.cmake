# Stage 3 instrumented:
# * Build an instrumented clang, so we can generate profile data for stage 3.


set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_CMAKE_ARGS -C ${CMAKE_CURRENT_LIST_DIR}/stage3.cmake CACHE BOOL "")
set(CLANG_BOOTSTRAP_TARGETS clang check-all distribution install-distribution clang-bolt CACHE BOOL "")
set(CLANG_BOLT OFF CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/stage3.cmake)
