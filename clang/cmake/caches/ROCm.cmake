# This file sets up a CMakeCache for the ROCm LLVM toolchain shipped by TheRock.
#
# Everything is configured in a single build. What is shipped is split into LLVM
# distributions, each built and installed by install-<name>-distribution:
#
#   compiler   The minimum to compile programs: clang, lld, the offload tools
#              the driver runs, and the host builtins.
#   runtimes   The host and GPU runtime libraries: compiler-rt, OpenMP, offload,
#              libc, libc++, libc++abi, and libunwind.
#   fortran    Flang, flang-rt, the Fortran modules for OpenMP, and the Flang
#              and MLIR tools, headers, libraries, and CMake packages. The
#              packages need dev.
#   dev        The LLVM, Clang, and LLD headers, libraries, and CMake packages,
#              llvm-config, the tablegens, and libclang.
#   tools      LLVM binutils such as llvm-objdump and opt, and Clang tooling
#              such as clang-tidy.
#   utils      LLVM test utilities such as FileCheck and not.
#   multilibs  Debug and AddressSanitizer builds of the OpenMP and offload
#              runtimes.
#
# Options read by this file must be passed before -C:
#   LLVM_INCLUDE_TESTS  Builds every LLVM and Clang tool which the tests need.

get_filename_component(_ROCM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# Before the compiler is detected, get_host_triple falls back to config.guess,
# which cannot run on Windows. The triple is only needed for the runtimes, which
# Windows builds for the default target.
if(LLVM_HOST_TRIPLE)
  set(ROCM_HOST_TRIPLE "${LLVM_HOST_TRIPLE}")
elseif(NOT CMAKE_HOST_WIN32)
  set(LLVM_MAIN_SRC_DIR "${_ROCM_ROOT}/llvm")
  include(${LLVM_MAIN_SRC_DIR}/cmake/modules/GetHostTriple.cmake)
  get_host_triple(ROCM_HOST_TRIPLE)
  unset(LLVM_MAIN_SRC_DIR)
endif()
set(ROCM_DEVICE_TRIPLE amdgpu-amd-amdhsa)

set(LLVM_TARGETS_TO_BUILD "Native;AMDGPU;SPIRV" CACHE STRING "")
if(CMAKE_HOST_WIN32)
  set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;lld" CACHE STRING "")
  set(_ROCM_HOST_RUNTIMES "compiler-rt")
else()
  set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra;mlir;flang;lld" CACHE STRING "")
  set(_ROCM_HOST_RUNTIMES "compiler-rt;libunwind;libcxx;libcxxabi;openmp;offload;flang-rt")
endif()

set(_ROCM_DEVICE_RUNTIMES "compiler-rt;libc;libcxx;libcxxabi;openmp;flang-rt")
set(LLVM_ENABLE_RUNTIMES "${_ROCM_HOST_RUNTIMES}" CACHE STRING "")

# General options.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(PACKAGE_VENDOR AMD CACHE STRING "")

set(LLVM_APPEND_VC_REV ON CACHE BOOL "")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "")
set(LLVM_ENABLE_Z3_SOLVER OFF CACHE BOOL "")
set(LLVM_ENABLE_ZLIB FORCE_ON CACHE STRING "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_OPTIMIZED_TABLEGEN ON CACHE BOOL "")

if(NOT CMAKE_HOST_WIN32)
  set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
  set(LLVM_ENABLE_LIBCXX ON CACHE BOOL "")
  set(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR ON CACHE BOOL "")
  set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "")

  # Installed to <rocm>/lib/llvm. The second entry reaches <rocm>/lib and the
  # third reaches the bundled system dependencies in <rocm>/lib/rocm_sysdeps.
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../../../lib;$ORIGIN/../../rocm_sysdeps/lib" CACHE STRING "")
endif()

# Tools.
#
# Clang and LLVM tools required by the toolchain.
set(_ROCM_LLVM_TOOLS
  llvm-ar                # llvm-ar, llvm-ranlib, llvm-lib, llvm-dlltool
  llvm-as
  llvm-config
  llvm-cov
  llvm-cxxfilt
  llvm-dis
  llvm-dwarfdump
  llvm-link
  llvm-mc
  llvm-nm
  llvm-objcopy           # llvm-objcopy, llvm-strip, llvm-bitcode-strip,
                         # llvm-install-name-tool, llvm-extract-bundle-entry
  llvm-objdump           # llvm-objdump, llvm-otool
  llvm-offload-binary    # llvm-offload-binary, clang-offload-packager
  llvm-profdata
  llvm-readobj           # llvm-readobj, llvm-readelf
  llvm-shlib             # libLLVM
  llvm-symbolizer        # llvm-symbolizer, llvm-addr2line
  opt
  yaml2obj)
set(_ROCM_CLANG_TOOLS
  clang-linker-wrapper
  clang-offload-bundler
  clang-shlib            # libclang-cpp
  clang-scan-deps
  driver                 # clang
  libclang
  offload-arch)          # offload-arch, amdgpu-arch
if(NOT LLVM_INCLUDE_TESTS)
  foreach(_project LLVM CLANG)
    string(TOLOWER ${_project} _dir)
    file(GLOB _tool_dirs LIST_DIRECTORIES true "${_ROCM_ROOT}/${_dir}/tools/*")
    foreach(_tool_dir ${_tool_dirs})
      if(NOT EXISTS "${_tool_dir}/CMakeLists.txt")
        continue()
      endif()
      get_filename_component(_tool ${_tool_dir} NAME)
      string(TOUPPER ${_tool} _option)
      string(REPLACE "-" "_" _option ${_option})
      # IN_LIST is unavailable here, as no policies are set yet.
      list(FIND _ROCM_${_project}_TOOLS ${_tool} _index)
      if(_index GREATER -1)
        set(${_project}_TOOL_${_option}_BUILD ON CACHE BOOL "")
      else()
        set(${_project}_TOOL_${_option}_BUILD OFF CACHE BOOL "")
      endif()
    endforeach()
  endforeach()
endif()

# Clang options.
set(CLANG_DEFAULT_LINKER lld CACHE STRING "")
set(CLANG_DEFAULT_RTLIB compiler-rt CACHE STRING "")
set(CLANG_DEFAULT_UNWINDLIB libgcc CACHE STRING "")
set(CLANG_ENABLE_CLANGD OFF CACHE BOOL "")
set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")
set(CLANG_TIDY_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")

# Flang options. Flang and the host flang-rt must agree on this. ppc64le has a
# native 128-bit long double and does not need libquadmath.
if(NOT CMAKE_HOST_WIN32 AND NOT ROCM_HOST_TRIPLE MATCHES "^(powerpc64le|ppc64le)")
  set(FLANG_RUNTIME_F128_MATH_LIB libquadmath CACHE STRING "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_FLANG_RUNTIME_F128_MATH_LIB libquadmath CACHE STRING "")
endif()

# Runtimes.
#
# Each enabled runtime already has a <runtime>-<triple> install component. This
# list adds the extra components the runtimes install separately, such as
# headers and Fortran modules.
if(CMAKE_HOST_WIN32)
  set(LLVM_RUNTIME_DISTRIBUTION_COMPONENTS compiler-rt-headers CACHE STRING "")
else()
  set(LLVM_BUILTIN_TARGETS "${ROCM_HOST_TRIPLE};${ROCM_DEVICE_TRIPLE}" CACHE STRING "")
  set(LLVM_RUNTIME_TARGETS "${ROCM_HOST_TRIPLE};${ROCM_DEVICE_TRIPLE}" CACHE STRING "")
  set(LLVM_RUNTIME_DISTRIBUTION_COMPONENTS
    compiler-rt-headers
    flang-rt-headers
    flang-rt-mod
    libomp-mod
    CACHE STRING "")

  # Host runtimes. libc++ is built static and not installed; libc++abi is shipped.
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LLVM_ENABLE_RUNTIMES "${_ROCM_HOST_RUNTIMES}" CACHE STRING "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBCXX_ENABLE_SHARED OFF CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBCXX_ENABLE_STATIC ON CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBCXX_INSTALL_HEADERS OFF CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBCXX_INSTALL_LIBRARY OFF CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBCXXABI_ENABLE_SHARED OFF CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBCXXABI_ENABLE_STATIC ON CACHE BOOL "")
  # Keeps libomp from copying build outputs into the source tree.
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBOMP_COPY_EXPORTS OFF CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBOMPTARGET_ENABLE_DEBUG ON CACHE BOOL "")

  # Device runtimes.
  set(RUNTIMES_${ROCM_DEVICE_TRIPLE}_LLVM_ENABLE_RUNTIMES "${_ROCM_DEVICE_RUNTIMES}" CACHE STRING "")
  set(RUNTIMES_${ROCM_DEVICE_TRIPLE}_CACHE_FILES
    "${_ROCM_ROOT}/compiler-rt/cmake/caches/AMDGPU.cmake;${_ROCM_ROOT}/libcxx/cmake/caches/AMDGPU.cmake" CACHE STRING "")
  set(RUNTIMES_${ROCM_DEVICE_TRIPLE}_FLANG_RT_LIBC_PROVIDER llvm CACHE STRING "")
  set(RUNTIMES_${ROCM_DEVICE_TRIPLE}_FLANG_RT_LIBCXX_PROVIDER llvm CACHE STRING "")

  # Runtime multilibs.
  #
  # Each <triple>+<name> is an extra build of that target's runtimes, installed
  # to lib/<triple>/<name>. It inherits every RUNTIMES_<triple>_* option; the
  # RUNTIMES_<triple>+<name>_* options below take precedence.
  set(LLVM_RUNTIME_MULTILIBS "debug;asan;debug+asan" CACHE STRING "")
  set(LLVM_RUNTIME_MULTILIB_debug_TARGETS "${ROCM_HOST_TRIPLE};${ROCM_DEVICE_TRIPLE}" CACHE STRING "")
  set(LLVM_RUNTIME_MULTILIB_asan_TARGETS "${ROCM_HOST_TRIPLE}" CACHE STRING "")
  set(LLVM_RUNTIME_MULTILIB_debug+asan_TARGETS "${ROCM_HOST_TRIPLE}" CACHE STRING "")

  foreach(_multilib debug asan debug+asan)
    set(RUNTIMES_${ROCM_HOST_TRIPLE}+${_multilib}_LLVM_ENABLE_RUNTIMES "openmp;offload" CACHE STRING "")
  endforeach()
  foreach(_multilib debug debug+asan)
    set(RUNTIMES_${ROCM_HOST_TRIPLE}+${_multilib}_CMAKE_BUILD_TYPE Debug CACHE STRING "")
  endforeach()
  # Sanitized builds use the base target's compiler-rt and leave operator new and
  # delete to the ASan runtime.
  foreach(_multilib asan debug+asan)
    set(RUNTIMES_${ROCM_HOST_TRIPLE}+${_multilib}_LLVM_USE_SANITIZER Address CACHE STRING "")
    set(RUNTIMES_${ROCM_HOST_TRIPLE}+${_multilib}_LLVM_BUILD_COMPILER_RT OFF CACHE BOOL "")
    set(RUNTIMES_${ROCM_HOST_TRIPLE}+${_multilib}_LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS OFF CACHE BOOL "")
    set(RUNTIMES_${ROCM_HOST_TRIPLE}+${_multilib}_LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS OFF CACHE BOOL "")
  endforeach()

  set(RUNTIMES_${ROCM_DEVICE_TRIPLE}+debug_LLVM_ENABLE_RUNTIMES "openmp" CACHE STRING "")
  set(RUNTIMES_${ROCM_DEVICE_TRIPLE}+debug_CMAKE_BUILD_TYPE Debug CACHE STRING "")
endif()

# Distributions.
#
# Runtime components are named <runtime>-<triple>, or just <runtime> on
# Windows. Targets that are not in any distribution are not installed and are
# omitted from the exported CMake packages. Each distribution installs its own
# exported targets, in the <project>-<name>-cmake-exports components, and the
# packages load them in this order, so a distribution must come after those it
# links against.
if(CMAKE_HOST_WIN32)
  set(LLVM_DISTRIBUTIONS "compiler;runtimes;dev;tools;utils" CACHE STRING "")
else()
  set(LLVM_DISTRIBUTIONS "compiler;runtimes;fortran;dev;tools;utils;multilibs" CACHE STRING "")
endif()

# The tools the HIP driver runs and the libraries they link. Linux links with
# compiler-rt by default.
set(_ROCM_COMPILER_COMPONENTS
  clang
  clang-compiler-cmake-exports
  clang-linker-wrapper
  clang-offload-bundler
  clang-resource-headers
  compiler-cmake-exports
  lld
  lld-compiler-cmake-exports
  llvm-offload-binary
  offload-arch)
if(CMAKE_HOST_WIN32)
  list(APPEND _ROCM_COMPILER_COMPONENTS
    builtins)
else()
  list(APPEND _ROCM_COMPILER_COMPONENTS
    builtins-${ROCM_HOST_TRIPLE}
    clang-cpp
    LLVM)
endif()
set(LLVM_compiler_DISTRIBUTION_COMPONENTS ${_ROCM_COMPILER_COMPONENTS} CACHE STRING "")

if(CMAKE_HOST_WIN32)
  set(LLVM_runtimes_DISTRIBUTION_COMPONENTS
    compiler-rt
    compiler-rt-headers
    CACHE STRING "")
else()
  set(LLVM_runtimes_DISTRIBUTION_COMPONENTS
    compiler-rt-${ROCM_HOST_TRIPLE}
    compiler-rt-headers-${ROCM_HOST_TRIPLE}
    cxxabi-${ROCM_HOST_TRIPLE}
    offload-${ROCM_HOST_TRIPLE}
    openmp-${ROCM_HOST_TRIPLE}
    unwind-${ROCM_HOST_TRIPLE}
    builtins-${ROCM_DEVICE_TRIPLE}
    compiler-rt-${ROCM_DEVICE_TRIPLE}
    cxx-${ROCM_DEVICE_TRIPLE}
    cxxabi-${ROCM_DEVICE_TRIPLE}
    libc-${ROCM_DEVICE_TRIPLE}
    openmp-${ROCM_DEVICE_TRIPLE}
    CACHE STRING "")

  set(LLVM_fortran_DISTRIBUTION_COMPONENTS
    flang
    flang-fortran-binding
    MLIR                 # libMLIR, which flang links against
    flang-rt-${ROCM_HOST_TRIPLE}
    flang-rt-${ROCM_DEVICE_TRIPLE}
    flang-rt-mod-${ROCM_HOST_TRIPLE}
    flang-rt-mod-${ROCM_DEVICE_TRIPLE}
    libomp-mod-${ROCM_HOST_TRIPLE}
    libomp-mod-${ROCM_DEVICE_TRIPLE}
    # Development files.
    flang-cmake-exports
    flang-fortran-cmake-exports
    flang-headers
    flang-libraries
    flang-rt-headers-${ROCM_HOST_TRIPLE}
    mlir-cmake-exports
    mlir-fortran-cmake-exports
    mlir-headers
    mlir-libraries
    # Flang and MLIR tools.
    bbc
    f18-parse-demo
    fir-lsp-server
    fir-opt
    tco
    mlir-irdl-to-cpp
    mlir-linalg-ods-yaml-gen
    mlir-lsp-server
    mlir-opt
    mlir-pdll
    mlir-pdll-lsp-server
    mlir-query
    mlir-reduce
    mlir-rewrite
    mlir-runner
    mlir-src-sharder
    mlir-tblgen
    mlir-translate
    tblgen-lsp-server
    tblgen-to-irdl
    CACHE STRING "")
endif()

set(LLVM_dev_DISTRIBUTION_COMPONENTS
  cmake-exports
  dev-cmake-exports
  llvm-config
  llvm-headers
  llvm-libraries
  llvm-tblgen
  clang-cmake-exports
  clang-dev-cmake-exports
  clang-headers
  clang-libraries
  clang-tblgen
  libclang
  libclang-headers
  lld-cmake-exports
  lld-dev-cmake-exports
  lld-headers
  lld-libraries
  CACHE STRING "")

set(_ROCM_TOOLS_COMPONENTS
  clang-offload-packager
  hlsl-resource-headers
  llvm-addr2line
  llvm-ar
  llvm-as
  llvm-bitcode-strip
  llvm-cov
  llvm-cxxfilt
  llvm-dis
  llvm-dlltool
  llvm-dwarfdump
  llvm-extract-bundle-entry
  llvm-install-name-tool
  llvm-lib
  llvm-link
  llvm-mc
  llvm-nm
  llvm-objcopy
  llvm-objdump
  llvm-otool
  llvm-profdata
  llvm-ranlib
  llvm-readelf
  llvm-readobj
  llvm-strip
  llvm-symbolizer
  opt
  tools-cmake-exports
  # clang-tools-extra.
  clang-apply-replacements
  clang-change-namespace
  clang-doc
  clang-include-cleaner
  clang-include-fixer
  clang-move
  clang-query
  clang-reorder-fields
  clang-tidy
  clang-tidy-headers
  clang-tools-cmake-exports
  find-all-symbols
  hmaptool
  modularize
  pp-trace)
if(CMAKE_HOST_WIN32)
  list(APPEND _ROCM_TOOLS_COMPONENTS
    clang-scan-deps)
endif()
set(LLVM_tools_DISTRIBUTION_COMPONENTS ${_ROCM_TOOLS_COMPONENTS} CACHE STRING "")

set(LLVM_utils_DISTRIBUTION_COMPONENTS
  count
  FileCheck
  llvm-PerfectShuffle
  llvm-test-mustache-spec
  not
  split-file
  UnicodeNameMappingGenerator
  utils-cmake-exports
  yaml-bench
  yaml2obj
  CACHE STRING "")

if(NOT CMAKE_HOST_WIN32)
  set(LLVM_multilibs_DISTRIBUTION_COMPONENTS
    offload-${ROCM_HOST_TRIPLE}+asan
    offload-${ROCM_HOST_TRIPLE}+debug
    offload-${ROCM_HOST_TRIPLE}+debug+asan
    openmp-${ROCM_HOST_TRIPLE}+asan
    openmp-${ROCM_HOST_TRIPLE}+debug
    openmp-${ROCM_HOST_TRIPLE}+debug+asan
    openmp-${ROCM_DEVICE_TRIPLE}+debug
    CACHE STRING "")
endif()
