# Two-stage build of Flang with the 2nd stage optimized using BOLT

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CLANG_BOLT "INSTRUMENT" CACHE STRING "")
set(FLANG_BOLT ${CLANG_BOLT} CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--emit-relocs,-znow" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--emit-relocs,-znow" CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "bolt;clang;flang" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES "compiler-rt;flang-rt;libunwind;openmp" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")

# setup toolchain
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  clang-resource-headers
  flang
  runtimes
  CACHE STRING "")
