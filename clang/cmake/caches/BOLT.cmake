set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CLANG_BOLT_INSTRUMENT ON CACHE BOOL "")
set(CLANG_BOLT_INSTRUMENT_PROJECTS "llvm" CACHE STRING "")
set(CLANG_BOLT_INSTRUMENT_TARGETS "count" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--emit-relocs,-znow" CACHE STRING "")
set(CLANG_BOLT_INSTRUMENT_EXTRA_CMAKE_FLAGS "" CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "bolt;clang" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")

# setup toolchain
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  clang-resource-headers
  CACHE STRING "")

# Disable function splitting enabled by default in GCC8+
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-reorder-blocks-and-partition")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-reorder-blocks-and-partition")
endif()
