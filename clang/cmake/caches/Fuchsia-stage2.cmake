# This file sets up a CMakeCache for the second stage of a Fuchsia toolchain
# build.

set(LLVM_TARGETS_TO_BUILD X86;ARM;AArch64 CACHE STRING "")

set(PACKAGE_VENDOR Fuchsia CACHE STRING "")

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_EXTERNALIZE_DEBUGINFO ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")

set(LLVM_ENABLE_LTO ON CACHE BOOL "")
if(NOT APPLE)
  set(LLVM_ENABLE_LLD ON CACHE BOOL "")
  set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
endif()

if(APPLE)
  set(LLDB_CODESIGN_IDENTITY "" CACHE STRING "")
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -gline-tables-only -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -gline-tables-only -DNDEBUG" CACHE STRING "")

set(LLVM_BUILTIN_TARGETS "x86_64-fuchsia;aarch64-fuchsia" CACHE STRING "")
foreach(target x86_64;aarch64)
  set(BUILTINS_${target}-fuchsia_CMAKE_SYSROOT ${FUCHSIA_${target}_SYSROOT} CACHE PATH "")
  set(BUILTINS_${target}-fuchsia_CMAKE_SYSTEM_NAME Fuchsia CACHE STRING "")
endforeach()

if(NOT APPLE)
  set(LLVM_BUILTIN_TARGETS "default;${LLVM_BUILTIN_TARGETS}" CACHE STRING "" FORCE)
endif()

set(LLVM_RUNTIME_TARGETS "default;x86_64-fuchsia;aarch64-fuchsia;x86_64-fuchsia-asan:x86_64-fuchsia;aarch64-fuchsia-asan:aarch64-fuchsia" CACHE STRING "")
foreach(target x86_64;aarch64)
  set(RUNTIMES_${target}-fuchsia_CMAKE_BUILD_WITH_INSTALL_RPATH ON CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_CMAKE_SYSROOT ${FUCHSIA_${target}_SYSROOT} CACHE PATH "")
  set(RUNTIMES_${target}-fuchsia_CMAKE_SYSTEM_NAME Fuchsia CACHE STRING "")
  set(RUNTIMES_${target}-fuchsia_UNIX 1 CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBUNWIND_ENABLE_STATIC OFF CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBCXXABI_ENABLE_STATIC OFF CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_LIBCXX_ABI_VERSION 2 CACHE STRING "")
  set(RUNTIMES_${target}-fuchsia_LIBCXX_ENABLE_STATIC OFF CACHE BOOL "")
  set(RUNTIMES_${target}-fuchsia_SANITIZER_USE_COMPILER_RT ON CACHE BOOL "")

  set(RUNTIMES_${target}-fuchsia-asan_LLVM_USE_SANITIZER Address CACHE STRING "")
  set(RUNTIMES_${target}-fuchsia-asan_LLVM_RUNTIMES_PREFIX "${target}-fuchsia/" CACHE STRING "")
  set(RUNTIMES_${target}-fuchsia-asan_LLVM_RUNTIMES_LIBDIR_SUFFIX "/asan" CACHE STRING "")
  set(RUNTIMES_${target}-fuchsia-asan_LIBCXX_INSTALL_HEADERS OFF CACHE BOOL "")
endforeach()

# Setup toolchain.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llc
  llvm-ar
  llvm-cov
  llvm-cxxfilt
  llvm-dwarfdump
  llvm-dsymutil
  llvm-lib
  llvm-nm
  llvm-objcopy
  llvm-objdump
  llvm-profdata
  llvm-ranlib
  llvm-readelf
  llvm-readobj
  llvm-size
  llvm-symbolizer
  opt
  sancov
  CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  lld
  lldb
  liblldb
  LTO
  clang-format
  clang-headers
  clang-tidy
  clangd
  builtins
  runtimes
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")
