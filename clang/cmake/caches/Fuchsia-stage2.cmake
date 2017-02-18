# This file sets up a CMakeCache for the second stage of a Fuchsia toolchain
# build.

set(LLVM_TARGETS_TO_BUILD X86;AArch64 CACHE STRING "")

set(PACKAGE_VENDOR Fuchsia CACHE STRING "")

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_TOOL_CLANG_TOOLS_EXTRA_BUILD OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_EXTERNALIZE_DEBUGINFO ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")

set(LLVM_ENABLE_LTO ON CACHE BOOL "")
if(NOT APPLE)
  set(LLVM_ENABLE_LLD ON CACHE BOOL "")
  set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -gline-tables-only -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -gline-tables-only -DNDEBUG" CACHE STRING "")

set(LLVM_BUILTIN_TARGETS "x86_64-fuchsia-none;aarch64-fuchsia-none" CACHE STRING "")
set(BUILTINS_x86_64-fuchsia-none_CMAKE_SYSROOT ${FUCHSIA_SYSROOT} CACHE STRING "")
set(BUILTINS_x86_64-fuchsia-none_CMAKE_SYSTEM_NAME Fuchsia CACHE STRING "")
set(BUILTINS_aarch64-fuchsia-none_CMAKE_SYSROOT ${FUCHSIA_SYSROOT} CACHE STRING "")
set(BUILTINS_aarch64-fuchsia-none_CMAKE_SYSTEM_NAME Fuchsia CACHE STRING "")

# Setup toolchain.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llvm-ar
  llvm-cov
  llvm-cxxfilt
  llvm-dwarfdump
  llvm-dsymutil
  llvm-lib
  llvm-nm
  llvm-objdump
  llvm-profdata
  llvm-ranlib
  llvm-readobj
  llvm-size
  llvm-symbolizer
  CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  lld
  lldb
  liblldb
  LTO
  clang-format
  clang-headers
  builtins-x86_64-fuchsia-none
  builtins-aarch64-fuchsia-none
  runtimes
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")
