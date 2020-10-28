set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "")
set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(LLVM_BUILD_EXTERNAL_COMPILER_RT ON CACHE BOOL "")
set(BOOTSTRAP_LLVM_ENABLE_LTO ON CACHE BOOL "")

# Use LLD do have less requirements on system linker, unless we're on an apple
# platform where the system compiler is to be prefered.
if(APPLE)
    set(BOOTSTRAP_LLVM_ENABLE_LLD OFF CACHE BOOL "")
else()
    set(BOOTSTRAP_LLVM_ENABLE_LLD ON CACHE BOOL "")
endif()


set(CLANG_BOOTSTRAP_TARGETS
  clang
  check-all
  check-llvm
  check-clang
  test-suite CACHE STRING "")

set(CLANG_BOOTSTRAP_CMAKE_ARGS
  -C ${CMAKE_CURRENT_LIST_DIR}/3-stage-base.cmake
  CACHE STRING "")
