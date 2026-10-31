set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "clang" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD "host" CACHE STRING "")

# The components we install to create the toolchain we benchmark.
set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  clang-resource-headers
  llvm-size   # used to measure code size
  llvm-nm     # needed by the microbenchmarks
  llvm-ar     # needed by the microbenchmarks
  llvm-ranlib # needed by the microbenchmarks
  CACHE STRING "")
