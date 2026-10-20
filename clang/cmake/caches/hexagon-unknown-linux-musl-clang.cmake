
set(LLVM_TARGETS_TO_BUILD "Hexagon" CACHE STRING "")
set(LLVM_DEFAULT_TARGET_TRIPLE "hexagon-unknown-linux-musl" CACHE STRING "")
set(CLANG_DEFAULT_CXX_STDLIB "libc++" CACHE STRING "")
set(CLANG_DEFAULT_OBJCOPY "llvm-objcopy" CACHE STRING "")
set(CLANG_DEFAULT_RTLIB "compiler-rt" CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB "libunwind" CACHE STRING "")
set(CLANG_DEFAULT_LINKER "lld" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "clang;lld" CACHE STRING "")

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
# Enabling toolchain-only causes problems when doing some of the
# subsequent builds, will need to investigate:
set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")
