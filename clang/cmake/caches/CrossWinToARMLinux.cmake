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
#       -DTOOLCHAIN_TARGET_TRIPLE=armv7-unknown-linux-gnueabihf ^
#       -DCMAKE_INSTALL_PREFIX=../install ^
#       -DDEFAULT_SYSROOT=<path-to-develop-arm-linux-root-fs> ^
#       -DLLVM_AR=<llvm_obj_root>/bin/llvm-ar[.exe] ^
#       -DCMAKE_CXX_FLAGS="-D__OPTIMIZE__" ^
#       -DREMOTE_TEST_HOST="<hostname>" ^
#       -DREMOTE_TEST_USER="<ssh_user_name>" ^
#       -C<llvm_src_root>/llvm-project/clang/cmake/caches/CrossWinToARMLinux.cmake ^
#       <llvm_src_root>/llvm-project/llvm
# Build:
#  cmake --build . --target install
# Tests:
#  cmake --build . --target check-llvm
#  cmake --build . --target check-clang
#  cmake --build . --target check-lld
#  cmake --build . --target check-compiler-rt-<TOOLCHAIN_TARGET_TRIPLE>
#  cmake --build . --target check-cxxabi-<TOOLCHAIN_TARGET_TRIPLE>
#  cmake --build . --target check-unwind-<TOOLCHAIN_TARGET_TRIPLE>
#  cmake --build . --target check-cxx-<TOOLCHAIN_TARGET_TRIPLE>
# (another way to execute the tests)
# python bin/llvm-lit.py -v --threads=32 runtimes/runtimes-<TOOLCHAIN_TARGET_TRIPLE>bins/libunwind/test  2>&1 | tee libunwind-tests.log
# python bin/llvm-lit.py -v --threads=32 runtimes/runtimes-<TOOLCHAIN_TARGET_TRIPLE>-bins/libcxxabi/test 2>&1 | tee libcxxabi-tests.log
# python bin/llvm-lit.py -v --threads=32 runtimes/runtimes-<TOOLCHAIN_TARGET_TRIPLE>-bins/libcxx/test    2>&1 | tee libcxx-tests.log


# LLVM_PROJECT_DIR is the path to the llvm-project directory.
# The right way to compute it would probably be to use "${CMAKE_SOURCE_DIR}/../",
# but CMAKE_SOURCE_DIR is set to the wrong value on earlier CMake versions
# that we still need to support (for instance, 3.10.2).
get_filename_component(LLVM_PROJECT_DIR
                       "${CMAKE_CURRENT_LIST_DIR}/../../../"
                       ABSOLUTE)

if (NOT DEFINED DEFAULT_SYSROOT)
  message(WARNING "DEFAULT_SYSROOT must be specified for the cross toolchain build.")
endif()

if (NOT DEFINED LLVM_ENABLE_ASSERTIONS)
  set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
endif()
if (NOT DEFINED LLVM_ENABLE_PROJECTS)
  set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;lld" CACHE STRING "")
endif()
if (NOT DEFINED LLVM_ENABLE_RUNTIMES)
  set(LLVM_ENABLE_RUNTIMES "compiler-rt;libunwind;libcxxabi;libcxx" CACHE STRING "")
endif()

if (NOT DEFINED TOOLCHAIN_TARGET_TRIPLE)
  set(TOOLCHAIN_TARGET_TRIPLE "aarch64-unknown-linux-gnu")
else()
  #NOTE: we must normalize specified target triple to a fully specified triple,
  # including the vendor part. It is necessary to synchronize the runtime library
  # installation path and operable target triple by Clang to get a correct runtime
  # path through `-print-runtime-dir` Clang option.
  string(REPLACE "-" ";" TOOLCHAIN_TARGET_TRIPLE "${TOOLCHAIN_TARGET_TRIPLE}")
  list(LENGTH TOOLCHAIN_TARGET_TRIPLE TOOLCHAIN_TARGET_TRIPLE_LEN)
  if (TOOLCHAIN_TARGET_TRIPLE_LEN LESS 3)
    message(FATAL_ERROR "invalid target triple")
  endif()
  # We suppose missed vendor's part.
  if (TOOLCHAIN_TARGET_TRIPLE_LEN LESS 4)
    list(INSERT TOOLCHAIN_TARGET_TRIPLE 1 "unknown")
  endif()
  string(REPLACE ";" "-" TOOLCHAIN_TARGET_TRIPLE "${TOOLCHAIN_TARGET_TRIPLE}")
endif()

message(STATUS "Toolchain target triple: ${TOOLCHAIN_TARGET_TRIPLE}")

if (NOT DEFINED LLVM_TARGETS_TO_BUILD)
  if ("${TOOLCHAIN_TARGET_TRIPLE}" MATCHES "^(armv|arm32)+")
    set(LLVM_TARGETS_TO_BUILD "ARM" CACHE STRING "")
  endif()
  if ("${TOOLCHAIN_TARGET_TRIPLE}" MATCHES "^(aarch64|arm64)+")
    set(LLVM_TARGETS_TO_BUILD "AArch64" CACHE STRING "")
  endif()
endif()

message(STATUS "Toolchain target to build: ${LLVM_TARGETS_TO_BUILD}")

if (NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
endif()

set(CMAKE_CROSSCOMPILING                    ON CACHE BOOL "")
set(CMAKE_CL_SHOWINCLUDES_PREFIX            "Note: including file: " CACHE STRING "")
# Required if COMPILER_RT_DEFAULT_TARGET_ONLY is ON
set(CMAKE_C_COMPILER_TARGET                 "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")
set(CMAKE_CXX_COMPILER_TARGET               "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")

set(LLVM_DEFAULT_TARGET_TRIPLE              "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")
set(LLVM_TARGET_ARCH                        "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")
set(LLVM_LIT_ARGS                           "-vv ${LLVM_LIT_ARGS}" CACHE STRING "" FORCE)

set(CLANG_DEFAULT_CXX_STDLIB                "libc++" CACHE STRING "")
set(CLANG_DEFAULT_LINKER                    "lld" CACHE STRING "")
set(CLANG_DEFAULT_OBJCOPY                   "llvm-objcopy" CACHE STRING "")
set(CLANG_DEFAULT_RTLIB                     "compiler-rt" CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB                 "libunwind" CACHE STRING "")

if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY            "MultiThreaded" CACHE STRING "")
endif()

# Set up RPATH for the target runtime/builtin libraries.
# See some details here: https://reviews.llvm.org/D91099
if (NOT DEFINED RUNTIMES_INSTALL_RPATH)
  set(RUNTIMES_INSTALL_RPATH                "\$ORIGIN/../lib;${CMAKE_INSTALL_PREFIX}/lib")
endif()

set(LLVM_BUILTIN_TARGETS                    "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")

set(BUILTINS_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_SYSTEM_NAME                         "Linux" CACHE STRING "")
set(BUILTINS_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_SYSROOT                             "${DEFAULT_SYSROOT}"  CACHE STRING "")
set(BUILTINS_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_INSTALL_RPATH                       "${RUNTIMES_INSTALL_RPATH}"  CACHE STRING "")
set(BUILTINS_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_BUILD_WITH_INSTALL_RPATH            ON  CACHE BOOL "")
set(BUILTINS_${TOOLCHAIN_TARGET_TRIPLE}_LLVM_CMAKE_DIR                            "${LLVM_PROJECT_DIR}/llvm/cmake/modules" CACHE PATH "")

set(LLVM_RUNTIME_TARGETS                    "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR      ON CACHE BOOL "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LLVM_ENABLE_RUNTIMES                      "${LLVM_ENABLE_RUNTIMES}" CACHE STRING "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_SYSTEM_NAME                         "Linux" CACHE STRING "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_SYSROOT                             "${DEFAULT_SYSROOT}"  CACHE STRING "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_INSTALL_RPATH                       "${RUNTIMES_INSTALL_RPATH}"  CACHE STRING "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_CMAKE_BUILD_WITH_INSTALL_RPATH            ON  CACHE BOOL "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_BUILTINS                ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_SANITIZERS              OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_XRAY                    OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_LIBFUZZER               OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_PROFILE                 OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_CRT                     ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_BUILD_ORC                     OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_DEFAULT_TARGET_ONLY           ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_INCLUDE_TESTS                 ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_CAN_EXECUTE_TESTS             ON CACHE BOOL "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_USE_BUILTINS_LIBRARY          ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_CXX_LIBRARY                   libcxx CACHE STRING "")
# Tell Clang to seach C++ headers alongside with the just-built binaries for the C++ compiler-rt tests.
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_TEST_COMPILER_CFLAGS          "--stdlib=libc++" CACHE STRING "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBUNWIND_USE_COMPILER_RT                 ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBUNWIND_ENABLE_SHARED                   OFF CACHE BOOL "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_USE_LLVM_UNWINDER               ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_ENABLE_STATIC_UNWINDER          ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_USE_COMPILER_RT                 ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS   OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_ENABLE_SHARED                   OFF CACHE BOOL "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_USE_COMPILER_RT                    ON CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_ENABLE_SHARED                      OFF CACHE BOOL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_ABI_VERSION                        2 CACHE STRING "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_CXX_ABI                            "libcxxabi" CACHE STRING "")    #!!!
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS      ON CACHE BOOL "")

# Avoid searching for the python3 interpreter during the runtimes configuration for the cross builds.
# It starts searching the python3 package using the target's sysroot path, that usually is not compatible with the build host.
find_package(Python3 COMPONENTS Interpreter)
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_Python3_EXECUTABLE                        ${Python3_EXECUTABLE} CACHE PATH "")

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBUNWIND_TEST_PARAMS_default "${RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_TEST_PARAMS}")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_TEST_PARAMS_default "${RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_TEST_PARAMS}")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_TEST_PARAMS_default "${RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_TEST_PARAMS}")

# Remote test configuration.
if(DEFINED REMOTE_TEST_HOST)
  # Allow override with the custom values.
  if(NOT DEFINED DEFAULT_TEST_EXECUTOR)
    set(DEFAULT_TEST_EXECUTOR                 "\\\"${Python3_EXECUTABLE}\\\" \\\"${LLVM_PROJECT_DIR}/libcxx/utils/ssh.py\\\" --host=${REMOTE_TEST_USER}@${REMOTE_TEST_HOST}")
  endif()

  set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_COMPILER_RT_EMULATOR
        "\\\"${Python3_EXECUTABLE}\\\" \\\"${LLVM_PROJECT_DIR}/llvm/utils/remote-exec.py\\\" --host=${REMOTE_TEST_USER}@${REMOTE_TEST_HOST}"
        CACHE STRING "")

  list(APPEND RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBUNWIND_TEST_PARAMS_default "executor=${DEFAULT_TEST_EXECUTOR}")
  list(APPEND RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_TEST_PARAMS_default "executor=${DEFAULT_TEST_EXECUTOR}")
  list(APPEND RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_TEST_PARAMS_default    "executor=${DEFAULT_TEST_EXECUTOR}")
endif()

set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBUNWIND_TEST_PARAMS "${RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBUNWIND_TEST_PARAMS_default}" CACHE INTERNAL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_TEST_PARAMS "${RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXXABI_TEST_PARAMS_default}" CACHE INTERNAL "")
set(RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_TEST_PARAMS "${RUNTIMES_${TOOLCHAIN_TARGET_TRIPLE}_LIBCXX_TEST_PARAMS_default}" CACHE INTERNAL "")

set(LLVM_INSTALL_TOOLCHAIN_ONLY ON CACHE BOOL "")
set(LLVM_TOOLCHAIN_TOOLS
  llvm-ar
  llvm-cov
  llvm-cxxfilt
  llvm-dwarfdump
  llvm-lib
  llvm-nm
  llvm-objdump
  llvm-pdbutil
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
