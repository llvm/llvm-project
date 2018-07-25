# This file sets up a CMakeCache for the second stage of a Fuchsia toolchain
# build.

set(LLVM_TARGETS_TO_BUILD X86;ARM;AArch64 CACHE STRING "")

set(PACKAGE_VENDOR Fuchsia CACHE STRING "")

set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES OFF CACHE BOOL "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB ON CACHE BOOL "")
set(LLVM_EXTERNALIZE_DEBUGINFO ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT OFF CACHE BOOL "")

set(LLVM_ENABLE_LTO ON CACHE BOOL "")
if(NOT APPLE)
  set(LLVM_ENABLE_LLD ON CACHE BOOL "")
  set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
  set(CLANG_DEFAULT_OBJCOPY llvm-objcopy CACHE STRING "")
endif()
set(CLANG_DEFAULT_CXX_STDLIB libc++ CACHE STRING "")
set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -gline-tables-only -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -gline-tables-only -DNDEBUG" CACHE STRING "")

if(APPLE)
  list(APPEND BUILTIN_TARGETS "default")
  list(APPEND RUNTIME_TARGETS "default")
elseif(UNIX)
  foreach(target i386;x86_64;armhf;aarch64)
    if(LINUX_${target}_SYSROOT)
      # Set the per-target builtins options.
      list(APPEND BUILTIN_TARGETS "${target}-linux-gnu")
      set(BUILTINS_${target}-linux-gnu_CMAKE_SYSTEM_NAME Linux CACHE STRING "")
      set(BUILTINS_${target}-linux-gnu_CMAKE_BUILD_TYPE Release CACHE STRING "")
      set(BUILTINS_${target}-linux-gnu_CMAKE_SYSROOT ${LINUX_${target}_SYSROOT} CACHE STRING "")

      # Set the per-target runtimes options.
      list(APPEND RUNTIME_TARGETS "${target}-linux-gnu")
      set(RUNTIMES_${target}-linux-gnu_CMAKE_SYSTEM_NAME Linux CACHE STRING "")
      set(RUNTIMES_${target}-linux-gnu_CMAKE_BUILD_TYPE Release CACHE STRING "")
      set(RUNTIMES_${target}-linux-gnu_CMAKE_SYSROOT ${LINUX_${target}_SYSROOT} CACHE STRING "")
      set(RUNTIMES_${target}-linux-gnu_LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBUNWIND_ENABLE_SHARED OFF CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBUNWIND_INSTALL_LIBRARY OFF CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXXABI_ENABLE_STATIC_UNWINDER ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXXABI_INSTALL_LIBRARY OFF CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_LIBCXX_ENABLE_STATIC_ABI_LIBRARY ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_SANITIZER_CXX_ABI "libc++" CACHE STRING "")
      set(RUNTIMES_${target}-linux-gnu_SANITIZER_CXX_ABI_INTREE ON CACHE BOOL "")
      set(RUNTIMES_${target}-linux-gnu_COMPILER_RT_USE_BUILTINS_LIBRARY ON CACHE BOOL "")
    endif()
  endforeach()
endif()

if(FUCHSIA_SDK)
  set(FUCHSIA_aarch64_NAME arm64)
  set(FUCHSIA_x86_64_NAME x64)
  foreach(target x86_64;aarch64)
    set(FUCHSIA_${target}_COMPILER_FLAGS "-I${FUCHSIA_SDK}/pkg/fdio/include")
    set(FUCHSIA_${target}_LINKER_FLAGS "-L${FUCHSIA_SDK}/arch/${FUCHSIA_${target}_NAME}/lib")
    set(FUCHSIA_${target}_SYSROOT "${FUCHSIA_SDK}/arch/${FUCHSIA_${target}_NAME}/sysroot")
  endforeach()

  foreach(target x86_64;aarch64)
    # Set the per-target builtins options.
    list(APPEND BUILTIN_TARGETS "${target}-fuchsia")
    set(BUILTINS_${target}-fuchsia_CMAKE_SYSTEM_NAME Fuchsia CACHE STRING "")
    set(BUILTINS_${target}-fuchsia_CMAKE_BUILD_TYPE Release CACHE STRING "")
    set(BUILTINS_${target}-fuchsia_CMAKE_ASM_FLAGS ${FUCHSIA_${target}_COMPILER_FLAGS} CACHE PATH "")
    set(BUILTINS_${target}-fuchsia_CMAKE_C_FLAGS ${FUCHSIA_${target}_COMPILER_FLAGS} CACHE PATH "")
    set(BUILTINS_${target}-fuchsia_CMAKE_CXX_FLAGS ${FUCHSIA_${target}_COMPILER_FLAGS} CACHE PATH "")
    set(BUILTINS_${target}-fuchsia_CMAKE_SHARED_LINKER_FLAGS ${FUCHSIA_${target}_LINKER_FLAGS} CACHE PATH "")
    set(BUILTINS_${target}-fuchsia_CMAKE_MODULE_LINKER_FLAGS ${FUCHSIA_${target}_LINKER_FLAGS} CACHE PATH "")
    set(BUILTINS_${target}-fuchsia_CMAKE_EXE_LINKER_FLAGS ${FUCHSIA_${target}_LINKER_FLAGS} CACHE PATH "")
    set(BUILTINS_${target}-fuchsia_CMAKE_SYSROOT ${FUCHSIA_${target}_SYSROOT} CACHE PATH "")

    # Set the per-target runtimes options.
    list(APPEND RUNTIME_TARGETS "${target}-fuchsia")
    set(RUNTIMES_${target}-fuchsia_CMAKE_SYSTEM_NAME Fuchsia CACHE STRING "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_BUILD_TYPE Release CACHE STRING "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_BUILD_WITH_INSTALL_RPATH ON CACHE STRING "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_ASM_FLAGS ${FUCHSIA_${target}_COMPILER_FLAGS} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_C_FLAGS ${FUCHSIA_${target}_COMPILER_FLAGS} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_CXX_FLAGS ${FUCHSIA_${target}_COMPILER_FLAGS} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_SHARED_LINKER_FLAGS ${FUCHSIA_${target}_LINKER_FLAGS} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_MODULE_LINKER_FLAGS ${FUCHSIA_${target}_LINKER_FLAGS} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_EXE_LINKER_FLAGS ${FUCHSIA_${target}_LINKER_FLAGS} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_CMAKE_SYSROOT ${FUCHSIA_${target}_SYSROOT} CACHE PATH "")
    set(RUNTIMES_${target}-fuchsia_LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBUNWIND_USE_COMPILER_RT ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBUNWIND_INSTALL_STATIC_LIBRARY OFF CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXXABI_USE_COMPILER_RT ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXXABI_USE_LLVM_UNWINDER ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXXABI_ENABLE_STATIC_UNWINDER ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXXABI_INSTALL_STATIC_LIBRARY OFF CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY OFF CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXX_USE_COMPILER_RT ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXX_ENABLE_STATIC_ABI_LIBRARY ON CACHE BOOL "")
    set(RUNTIMES_${target}-fuchsia_LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY OFF CACHE BOOL "")
  endforeach()

  set(LLVM_RUNTIME_SANITIZERS "Address" CACHE STRING "")
  set(LLVM_RUNTIME_SANITIZER_Address_TARGETS "x86_64-fuchsia;aarch64-fuchsia" CACHE STRING "")
endif()

set(LLVM_BUILTIN_TARGETS "${BUILTIN_TARGETS}" CACHE STRING "")
set(LLVM_RUNTIME_TARGETS "${RUNTIME_TARGETS}" CACHE STRING "")

# Setup toolchain.
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  dsymutil
  llc
  llvm-ar
  llvm-cov
  llvm-cxxfilt
  llvm-dwarfdump
  llvm-lib
  llvm-nm
  llvm-objcopy
  llvm-objdump
  llvm-profdata
  llvm-ranlib
  llvm-readelf
  llvm-readobj
  llvm-size
  llvm-strip
  llvm-symbolizer
  opt
  sancov
  CACHE STRING "")

set(LLVM_DISTRIBUTION_COMPONENTS
  clang
  libclang
  lld
  LTO
  clang-format
  clang-headers
  clang-include-fixer
  clang-refactor
  clang-tidy
  clangd
  builtins
  runtimes
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")
