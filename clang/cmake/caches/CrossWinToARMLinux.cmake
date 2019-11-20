# CrossWinToARMLinux.cmake
#
# Set up a CMakeCache for a cross Windows to ARM Linux toolchain build.
#
# This cache file can be used to build a cross toolchain to ARM Linux
# on Windows platform.
#
# NOTE: the build requires a development ARM Linux root filesystem to use
# proper target platform depended library and header files.
#
# Configure:
#  cmake -G Ninja ^
#       -DCMAKE_INSTALL_PREFIX=../install ^
#       -DDEFAULT_SYSROOT=<path-to-develop-arm-linux-root-fs> ^
#       -DLLVM_AR=<llvm_obj_root>/bin/llvm-ar[.exe] ^
#       -DCMAKE_CXX_FLAGS="-D__OPTIMIZE__" ^
#       -C<llvm_src_root>/llvm-project/clang/cmake/caches/CrossWinToARMLinux.cmake ^
#       <llvm_src_root>/llvm-project/llvm
# Build:
#  cmake --build . --target install
# Test:
#  cmake --build . --target check-llvm
#  cmake --build . --target check-clang
#  cmake --build . --target check-lld

if (NOT DEFINED DEFAULT_SYSROOT)
  message(WARNING "DEFAULT_SYSROOT must be specified for the cross toolchain build.")
endif()

if (DEFINED LLVM_AR)
  set(CMAKE_AR "${LLVM_AR}" CACHE STRING "")
endif()

if (NOT DEFINED LLVM_TARGETS_TO_BUILD)
  set(LLVM_TARGETS_TO_BUILD "ARM" CACHE STRING "")
endif()

if (NOT DEFINED CMAKE_C_COMPILER_TARGET)
  # Required if COMPILER_RT_DEFAULT_TARGET_ONLY is ON
  set(CMAKE_C_COMPILER_TARGET "armv7-linux-gnueabihf" CACHE STRING "")
endif()

if (NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
endif()

set(CMAKE_CROSSCOMPILING                    ON CACHE BOOL "")
set(CMAKE_CL_SHOWINCLUDES_PREFIX            "Note: including file: " CACHE STRING "")

set(LLVM_ENABLE_ASSERTIONS                  ON CACHE BOOL "")
set(LLVM_ENABLE_PROJECTS                    "clang;clang-tools-extra;lld" CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES                    "compiler-rt;libunwind;libcxxabi;libcxx" CACHE STRING "")
set(LLVM_DEFAULT_TARGET_TRIPLE              "${CMAKE_C_COMPILER_TARGET}" CACHE STRING "")
set(LLVM_TARGET_ARCH                        "${CMAKE_C_COMPILER_TARGET}" CACHE STRING "")
set(LLVM_LIT_ARGS                           "-vv ${LLVM_LIT_ARGS}" CACHE STRING "" FORCE)

set(CLANG_DEFAULT_LINKER                    "lld" CACHE STRING "")

set(COMPILER_RT_BUILD_BUILTINS              ON CACHE BOOL "")
set(COMPILER_RT_BUILD_SANITIZERS            OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_XRAY                  OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_LIBFUZZER             OFF CACHE BOOL "")
set(COMPILER_RT_BUILD_PROFILE               OFF CACHE BOOL "")
set(COMPILER_RT_DEFAULT_TARGET_ONLY         ON CACHE BOOL "")

set(LIBUNWIND_USE_COMPILER_RT               ON CACHE BOOL "")
set(LIBUNWIND_TARGET_TRIPLE                 "${CMAKE_C_COMPILER_TARGET}" CACHE STRING "")
set(LIBUNWIND_SYSROOT                       "${DEFAULT_SYSROOT}" CACHE STRING "")
set(LIBUNWIND_ENABLE_SHARED                 OFF CACHE BOOL "")

set(LIBCXXABI_USE_LLVM_UNWINDER             ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_STATIC_UNWINDER        ON CACHE BOOL "")
set(LIBCXXABI_USE_COMPILER_RT               ON CACHE BOOL "")
set(LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS OFF CACHE BOOL "")
set(LIBCXXABI_TARGET_TRIPLE                 "${CMAKE_C_COMPILER_TARGET}" CACHE STRING "")
set(LIBCXXABI_SYSROOT                       "${DEFAULT_SYSROOT}" CACHE STRING "")

set(LIBCXX_USE_COMPILER_RT                  ON CACHE BOOL "")
set(LIBCXX_TARGET_TRIPLE                    "${CMAKE_C_COMPILER_TARGET}" CACHE STRING "")
set(LIBCXX_SYSROOT                          "${DEFAULT_SYSROOT}" CACHE STRING "")

set(BUILTINS_CMAKE_ARGS                     "-DCMAKE_SYSTEM_NAME=Linux;-DCMAKE_AR=${CMAKE_AR}" CACHE STRING "")
set(RUNTIMES_CMAKE_ARGS                     "-DCMAKE_SYSTEM_NAME=Linux;-DCMAKE_AR=${CMAKE_AR}" CACHE STRING "")

set(LLVM_INSTALL_TOOLCHAIN_ONLY 			ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llvm-ar
  llvm-cov
  llvm-cxxfilt
  llvm-dwarfdump
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
  LTO
  clang-format
  builtins
  runtimes
  ${LLVM_TOOLCHAIN_TOOLS}
  CACHE STRING "")
