# Clang defaults for Hexagon Linux (musl) cross-compilation — distribution build.
#
# Used by build-toolchain.sh for the full install-distribution build path.
# Load this before hexagon-unknown-linux-musl-clang-dist.cmake.
#
# Sets the compiler driver defaults so that plain `clang` invocations
# target hexagon-unknown-linux-musl with the expected runtime libraries.


set(LLVM_DEFAULT_TARGET_TRIPLE "hexagon-unknown-linux-musl" CACHE STRING "")
set(CLANG_DEFAULT_CXX_STDLIB "libc++" CACHE STRING "")
set(CLANG_DEFAULT_RTLIB "compiler-rt" CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB "libunwind" CACHE STRING "")
set(CLANG_DEFAULT_LINKER "lld" CACHE STRING "")
set(CLANG_DEFAULT_OBJCOPY "llvm-objcopy" CACHE STRING "")
