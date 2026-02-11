# cross-linux-toolchain.cmake
#
# Set up a CMakeCache for a cross ARM Linux toolchain build on Windows/Linux hosts.
#
# This cache file can be used to build a multi-target cross toolchain to Linux
# on Windows/Linux platforms.
#
# NOTE: replaces CrossWinToARMLinux.cmake
#
# NOTE: the build requires a development ARM Linux root filesystem to use
# proper target platform depended library and header files.
#
# The build generates a proper clang configuration file with stored
# --sysroot argument for specified target triple. Also it is possible
# to specify configuration path via CMake arguments, such as
#   -DCLANG_CONFIG_FILE_USER_DIR=<full-path-to-clang-configs>
# and/or
#   -DCLANG_CONFIG_FILE_SYSTEM_DIR=<full-path-to-clang-configs>
#
# See more details here: https://clang.llvm.org/docs/UsersManual.html#configuration-files
#
# Configure:
#  cmake -G Ninja \
#       -DTOOLCHAIN_TARGET_TRIPLE=aarch64-unknown-linux-gnu \
#       -DTOOLCHAIN_TARGET_SYSROOTFS=<path-to-develop-arm-linux-root-fs> \
#       -DTOOLCHAIN_SHARED_LIBS=OFF \
#       -DCMAKE_INSTALL_PREFIX=../install \
#       -DCMAKE_CXX_FLAGS="-D__OPTIMIZE__" \
#       -DREMOTE_TEST_HOST="<hostname>" \
#       -DREMOTE_TEST_USER="<ssh_user_name>" \
#       -C <llvm_src_root>/llvm-project/clang/cmake/caches/cross-linux-toolchain.cmake \
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
#
# The custom parameters:
#   TOOLCHAIN_TARGET_TRIPLE (default: aarch64-unknown-linux-gnu)  - a list of target triples to support
#   TOOLCHAIN_TARGET_SYSROOTFS-<target>     - sysroot for the specific target triple (must be matched TOOLCHAIN_TARGET_TRIPLE)
#   TOOLCHAIN_TARGET_SYSROOTFS              - sysroot for other targets.
#   TOOLCHAIN_TARGET_COMPILER_FLAGS-<target>
#   TOOLCHAIN_TARGET_COMPILER_FLAGS
#   TOOLCHAIN_TARGET_LINKER_FLAGS-<target>
#   TOOLCHAIN_TARGET_LINKER_FLAGS
#   TOOLCHAIN_SHARED_LIBS (default: OFF)
#   TOOLCHAIN_STATIC_LIBS (default: NOT DEFINED)
#   TOOLCHAIN_USE_STATIC_LIBS (default: ON)
#


# LLVM_PROJECT_DIR is the path to the llvm-project directory.
# The right way to compute it would probably be to use "${CMAKE_SOURCE_DIR}/../",
# but CMAKE_SOURCE_DIR is set to the wrong value on earlier CMake versions
# that we still need to support (for instance, 3.10.2).
get_filename_component(LLVM_PROJECT_DIR
                       "${CMAKE_CURRENT_LIST_DIR}/../../../"
                       ABSOLUTE)
# Store the passed defs to use them later.
get_cmake_property(vars_ VARIABLES)

# Avoid searching for the python3 interpreter during the runtimes configuration for the cross builds.
# It starts searching the python3 package using the target's sysroot path, that usually is not compatible with the build host.
find_package(Python3 COMPONENTS Interpreter)

# Allow override with the custom values.
if(NOT DEFINED DEFAULT_TEST_EXECUTOR)
  set(DEFAULT_TEST_EXECUTOR                 "\\\"${Python3_EXECUTABLE}\\\" \\\"${LLVM_PROJECT_DIR}/libcxx/utils/ssh.py\\\" --host=${REMOTE_TEST_USER}@${REMOTE_TEST_HOST}")
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

#Note: this is a list by default
if (NOT DEFINED TOOLCHAIN_TARGET_TRIPLE)
  set(TOOLCHAIN_TARGET_TRIPLE "aarch64-unknown-linux-gnu")
endif()

#NOTE: we must normalize specified target triple to a fully specified triple,
# including the vendor part. It is necessary to synchronize the runtime library
# installation path and operable target triple by Clang to get a correct runtime
# path through `-print-runtime-dir` Clang option.
function(norm_triple name triple abi)
  string(REPLACE "-" ";" t_ "${name}")
  list(LENGTH t_ tlen_)
  if (tlen_ LESS 3)
    message(FATAL_ERROR "invalid target triple ${name}")
  endif()
  # We suppose missed vendor's part.
  if (tlen_ LESS 4)
    list(INSERT t_ 1 "unknown")
  endif()
  list(GET t_ 3 abi_)
  string(REPLACE ";" "-" t_ "${t_}")

  set(${triple} "${t_}" PARENT_SCOPE)
  set(${abi} ${abi_} PARENT_SCOPE)
endfunction()

message(STATUS "Toolchain target triples: ${TOOLCHAIN_TARGET_TRIPLE}")

# Build the shared libraries for libc++/libc++abi/libunwind.
if (NOT DEFINED TOOLCHAIN_SHARED_LIBS)
  set(TOOLCHAIN_SHARED_LIBS OFF)
endif()

# Enable usage of the static libunwind and libc++abi libraries.
if (NOT DEFINED TOOLCHAIN_USE_STATIC_LIBS)
  set(TOOLCHAIN_USE_STATIC_LIBS ON)
endif()

if (NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
endif()

if (NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY AND WIN32)
  #Note: Always specify MT DLL for the LLDB build configurations on Windows host.
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_MSVC_RUNTIME_LIBRARY            "MultiThreadedDebugDLL" CACHE STRING "")
  else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY            "MultiThreadedDLL" CACHE STRING "")
  endif()
  # Grab all ucrt/vcruntime related DLLs into the binary installation folder.
  set(CMAKE_INSTALL_UCRT_LIBRARIES          ON CACHE BOOL "")
endif()

# Set up RPATH for the target runtime/builtin libraries.
# See some details here: https://reviews.llvm.org/D91099
if (NOT DEFINED RUNTIMES_INSTALL_RPATH)
  set(RUNTIMES_INSTALL_RPATH                "\$ORIGIN/../lib;${CMAKE_INSTALL_PREFIX}/lib")
endif()

set(CMAKE_CL_SHOWINCLUDES_PREFIX            "Note: including file: " CACHE STRING "")

# Set the first target in the list as the default target.
# (Allow custom default targets)
if (NOT DEFINED LLVM_DEFAULT_TARGET_TRIPLE)
  list(GET TOOLCHAIN_TARGET_TRIPLE 0 LLVM_DEFAULT_TARGET_TRIPLE)
  set(LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}" CACHE STRING "")
endif()
string(REPLACE "-" ";" t_ "${LLVM_DEFAULT_TARGET_TRIPLE}")
list(GET t_ 0 LLVM_TARGET_ARCH)

# Clang configuration.
set(CLANG_DEFAULT_CXX_STDLIB                "libc++" CACHE STRING "")
set(CLANG_DEFAULT_LINKER                    "lld" CACHE STRING "")
set(CLANG_DEFAULT_OBJCOPY                   "llvm-objcopy" CACHE STRING "")
set(CLANG_DEFAULT_RTLIB                     "compiler-rt" CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB                 "libunwind" CACHE STRING "")

#
# Configure the builtin targets.
#
set(LLVM_BUILTIN_TARGETS                    "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")

foreach (target ${LLVM_BUILTIN_TARGETS})
  set(BUILTINS_${target}_CMAKE_SYSTEM_NAME                        "Linux" CACHE STRING "")
  set(BUILTINS_${target}_CMAKE_INSTALL_RPATH                      "${RUNTIMES_INSTALL_RPATH}"  CACHE STRING "")
  set(BUILTINS_${target}_CMAKE_BUILD_WITH_INSTALL_RPATH           ON  CACHE BOOL "")
  set(BUILTINS_${target}_LLVM_CMAKE_DIR                           "${LLVM_PROJECT_DIR}/llvm/cmake/modules" CACHE PATH "")

  if (DEFINED TOOLCHAIN_TARGET_COMPILER_FLAGS)
    foreach(lang C;CXX;ASM)
      set(BUILTINS_${target}_CMAKE_${lang}_FLAGS                  "${TOOLCHAIN_TARGET_COMPILER_FLAGS}" CACHE STRING "")
    endforeach()
  endif()
  foreach(type SHARED;MODULE;EXE)
    set(BUILTINS_${target}_CMAKE_${type}_LINKER_FLAGS             "-fuse-ld=lld" CACHE STRING "")
  endforeach()
endforeach()

#
# Configure all runtime targets.
#
set(LLVM_RUNTIME_TARGETS                    "${TOOLCHAIN_TARGET_TRIPLE}" CACHE STRING "")
set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR      ON CACHE BOOL "")

foreach(target ${LLVM_RUNTIME_TARGETS})
  norm_triple(${target} triple_ abi_)

  if (DEFINED "TOOLCHAIN_TARGET_SYSROOTFS-${target}")
    set(sysroot_ "${TOOLCHAIN_TARGET_SYSROOTFS-${target}}")
  elseif (DEFINED TOOLCHAIN_TARGET_SYSROOTFS)
    set(sysroot_ ${TOOLCHAIN_TARGET_SYSROOTFS})
  endif()

  file(REMOVE "${CMAKE_BINARY_DIR}/bin/${triple_}.cfg")

  if (DEFINED sysroot_)
    message(STATUS "Toolchain target sysroot: ${target}: ${sysroot_}")
    # Store the --sysroot argument for the compiler-rt test flags.
    set(sysroot_flags --sysroot='${sysroot_}')
    # Generate the clang configuration file for the specified target triple and store --sysroot in there.
    #Note: we use normalized target triple for the configuration file name.
    file(APPEND "${CMAKE_BINARY_DIR}/bin/${triple_}.cfg" ${sysroot_flags} "\n")
  endif()

  # Pass a list of enabled runtimes to the runtime.
  set(RUNTIMES_${target}_LLVM_ENABLE_RUNTIMES                     "${LLVM_ENABLE_RUNTIMES}" CACHE STRING "")

  set(RUNTIMES_${target}_CMAKE_SYSTEM_NAME                        "Linux" CACHE STRING "")
  set(RUNTIMES_${target}_CMAKE_INSTALL_RPATH                      "${RUNTIMES_INSTALL_RPATH}"  CACHE STRING "")
  set(RUNTIMES_${target}_CMAKE_BUILD_WITH_INSTALL_RPATH           ON  CACHE BOOL "")

  if (DEFINED TOOLCHAIN_TARGET_COMPILER_FLAGS OR DEFINED TOOLCHAIN_TARGET_COMPILER_FLAGS-${target})
    message(STATUS "Toolchain target compiler flags: ${target}: ${TOOLCHAIN_TARGET_COMPILER_FLAGS-${target}}")
    foreach(lang C;CXX;ASM)
      set(RUNTIMES_${target}_CMAKE_${lang}_FLAGS                  "${TOOLCHAIN_TARGET_COMPILER_FLAGS} ${TOOLCHAIN_TARGET_COMPILER_FLAGS-${target}}" CACHE STRING "")
    endforeach()
    # Update the target clang cofiguration file with these flags.
    file(APPEND "${CMAKE_BINARY_DIR}/bin/${triple_}.cfg" ${TOOLCHAIN_TARGET_COMPILER_FLAGS} "\n")
    file(APPEND "${CMAKE_BINARY_DIR}/bin/${triple_}.cfg" ${TOOLCHAIN_TARGET_COMPILER_FLAGS-${target}} "\n")
  endif()
  if (DEFINED TOOLCHAIN_TARGET_LINKER_FLAGS OR DEFINED TOOLCHAIN_TARGET_LINKER_FLAGS-${target})
    message(STATUS "Toolchain target linker flags: ${target}: ${TOOLCHAIN_TARGET_LINKER_FLAGS-${target}}")
    foreach(type SHARED;MODULE;EXE)
      set(RUNTIMES_${target}_CMAKE_${type}_LINKER_FLAGS             "-fuse-ld=lld ${TOOLCHAIN_TARGET_LINKER_FLAGS} ${TOOLCHAIN_TARGET_LINKER_FLAGS-${target}}" CACHE STRING "")
    endforeach()
  endif()

  if (abi_ MATCHES "(musl|pauthtest)")
    set(RUNTIMES_${target}_LIBCXX_HAS_MUSL_LIBC                   ON CACHE BOOL "")
    set(RUNTIMES_${target}_COMPILER_RT_BUILD_BUILTINS             OFF CACHE BOOL "")
  else()
    set(RUNTIMES_${target}_COMPILER_RT_BUILD_BUILTINS             ON CACHE BOOL "")
  endif()

  set(RUNTIMES_${target}_COMPILER_RT_USE_BUILTINS_LIBRARY         ON CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_CRT                    ON CACHE BOOL "")

  #Note: COMPILER_RT_DEFAULT_TARGET_ONLY must be off for COMPILER_RT_DEFAULT_TARGET_TRIPLE.
  set(RUNTIMES_${target}_COMPILER_RT_DEFAULT_TARGET_ONLY          ON CACHE BOOL "")

  # Required if COMPILER_RT_DEFAULT_TARGET_ONLY is ON
  if (RUNTIMES_${target}_COMPILER_RT_DEFAULT_TARGET_ONLY)
    set(RUNTIMES_${target}_CMAKE_C_COMPILER_TARGET                 "${target}" CACHE STRING "")
    set(RUNTIMES_${target}_CMAKE_CXX_COMPILER_TARGET               "${target}" CACHE STRING "")
  endif()

  set(RUNTIMES_${target}_COMPILER_RT_CXX_LIBRARY                  libcxx CACHE STRING "")

  set(RUNTIMES_${target}_COMPILER_RT_BUILD_SANITIZERS             OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_XRAY                   OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_LIBFUZZER              OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_PROFILE                OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_MEMPROF                OFF CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_BUILD_ORC                    OFF CACHE BOOL "")

  set(RUNTIMES_${target}_COMPILER_RT_INCLUDE_TESTS                ON CACHE BOOL "")
  set(RUNTIMES_${target}_COMPILER_RT_CAN_EXECUTE_TESTS            ON CACHE BOOL "")
  # The compiler-rt tests disable the clang configuration files during the execution by setting CLANG_NO_DEFAULT_CONFIG=1
  # and drops out the --sysroot from there. Provide it explicity via the test flags here if target sysroot has been specified.
  set(RUNTIMES_${target}_COMPILER_RT_TEST_COMPILER_CFLAGS          "--stdlib=libc++ ${sysroot_flags}" CACHE STRING "")

  set(RUNTIMES_${target}_LIBUNWIND_USE_COMPILER_RT                 ON CACHE BOOL "")

  set(RUNTIMES_${target}_LIBCXXABI_USE_LLVM_UNWINDER               ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_USE_COMPILER_RT                 ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS   OFF CACHE BOOL "")

  set(RUNTIMES_${target}_LIBCXX_USE_COMPILER_RT                    ON CACHE BOOL "")
  set(RUNTIMES_${target}_LIBCXX_CXX_ABI                            "libcxxabi" CACHE STRING "")    #!!!
  set(RUNTIMES_${target}_LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS      ON CACHE BOOL "")


  if (DEFINED TOOLCHAIN_SHARED_LIBS)
    set(RUNTIMES_${target}_LIBUNWIND_ENABLE_SHARED                   ${TOOLCHAIN_SHARED_LIBS} CACHE BOOL "")
    set(RUNTIMES_${target}_LIBCXXABI_ENABLE_SHARED                   ${TOOLCHAIN_SHARED_LIBS} CACHE BOOL "")
    set(RUNTIMES_${target}_LIBCXX_ENABLE_SHARED                      ${TOOLCHAIN_SHARED_LIBS} CACHE BOOL "")
  endif()
  if (DEFINED TOOLCHAIN_STATIC_LIBS)
    set(RUNTIMES_${target}_LIBUNWIND_ENABLE_STATIC                   ${TOOLCHAIN_STATIC_LIBS} CACHE BOOL "")
    set(RUNTIMES_${target}_LIBCXXABI_ENABLE_STATIC                   ${TOOLCHAIN_STATIC_LIBS} CACHE BOOL "")
    set(RUNTIMES_${target}_LIBCXX_ENABLE_STATIC                      ${TOOLCHAIN_STATIC_LIBS} CACHE BOOL "")
  endif()

  if (DEFINED TOOLCHAIN_USE_STATIC_LIBS)
    set(RUNTIMES_${target}_LIBCXXABI_ENABLE_STATIC_UNWINDER          ${TOOLCHAIN_USE_STATIC_LIBS} CACHE BOOL "")
    # Merge libc++ and libc++abi libraries into the single libc++ library file.
    set(RUNTIMES_${target}_LIBCXX_ENABLE_STATIC_ABI_LIBRARY          ${TOOLCHAIN_USE_STATIC_LIBS} CACHE BOOL "")
  endif()

  # Forcely disable the libc++ benchmarks on Windows build hosts
  # (current benchmark test configuration does not support the cross builds there).
  if (WIN32)
    set(RUNTIMES_${target}_LIBCXX_INCLUDE_BENCHMARKS               OFF CACHE BOOL "")
  endif(WIN32)

  set(RUNTIMES_${target}_Python3_EXECUTABLE                        ${Python3_EXECUTABLE} CACHE PATH "")

  set(RUNTIMES_${target}_LIBUNWIND_TEST_PARAMS_default "${RUNTIMES_${target}_TEST_PARAMS}")
  set(RUNTIMES_${target}_LIBCXXABI_TEST_PARAMS_default "${RUNTIMES_${target}_TEST_PARAMS}")
  set(RUNTIMES_${target}_LIBCXX_TEST_PARAMS_default "${RUNTIMES_${target}_TEST_PARAMS}")

  # Remote test configuration.
  if(DEFINED REMOTE_TEST_HOST)
    set(RUNTIMES_${target}_COMPILER_RT_EMULATOR
          "\\\"${Python3_EXECUTABLE}\\\" \\\"${LLVM_PROJECT_DIR}/llvm/utils/remote-exec.py\\\" --host=${REMOTE_TEST_USER}@${REMOTE_TEST_HOST}"
          CACHE STRING "")

    list(APPEND RUNTIMES_${target}_LIBUNWIND_TEST_PARAMS_default "executor=${DEFAULT_TEST_EXECUTOR}")
    list(APPEND RUNTIMES_${target}_LIBCXXABI_TEST_PARAMS_default "executor=${DEFAULT_TEST_EXECUTOR}")
    list(APPEND RUNTIMES_${target}_LIBCXX_TEST_PARAMS_default    "executor=${DEFAULT_TEST_EXECUTOR}")
  endif()

  set(RUNTIMES_${target}_LIBUNWIND_TEST_PARAMS "${RUNTIMES_${target}_LIBUNWIND_TEST_PARAMS_default}" CACHE INTERNAL "")
  set(RUNTIMES_${target}_LIBCXXABI_TEST_PARAMS "${RUNTIMES_${target}_LIBCXXABI_TEST_PARAMS_default}" CACHE INTERNAL "")
  set(RUNTIMES_${target}_LIBCXX_TEST_PARAMS "${RUNTIMES_${target}_LIBCXX_TEST_PARAMS_default}" CACHE INTERNAL "")

  # Apply all passed LIBCXX|LIBCXXABI|LIBUNWIND|COMPILER_RT parameters to each runtime targets.
  # Override the existing variable values by using FORCE.
  # Because we don't know a type of the passed vars, use INTERNAL keyword for that.
  foreach(v_ ${vars_})
    if(v_ MATCHES "^(LIBCXX|LIBCXXABI|LIBUNWIND|COMPILER_RT)_")
      set(RUNTIMES_${target}_${v_} ${${v_}} CACHE INTERNAL "" FORCE)
    endif()
  endforeach()
  unset(sysroot_)
  unset(sysroot_flags)
endforeach()

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
