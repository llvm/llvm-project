# Stage 2:
# * This is the final stage.
# * The goal is to have a clang that is LTO, PGO, and bolt optimized and also
#   statically linked to libcxx and compiler-rt.

set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES compiler-rt libcxx libcxxabi libunwind CACHE STRING "" FORCE)
set(LLVM_ENABLE_PROJECTS clang lld bolt CACHE STRING "" FORCE)
set(LLVM_ENABLE_LLD ON CACHE BOOL "")
set(LLVM_ENABLE_LTO THIN CACHE STRING "")
set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
set(LLVM_STATIC_LINK_CXX_STDLIB ON CACHE BOOL "")
set(CLANG_BOLT "INSTRUMENT" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--emit-relocs,-znow -rtlib=compiler-rt --unwindlib=libunwind -static-libgcc" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS "-rtlib=compiler-rt --unwindlib=libunwind -static-libgcc" CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS "-rtlib=compiler-rt --unwindlib=libunwind -static-libgcc" CACHE STRING "")
set(LLVM_DISTRIBUTION_COMPONENTS clang lld runtimes clang-resource-headers CACHE STRING "")
