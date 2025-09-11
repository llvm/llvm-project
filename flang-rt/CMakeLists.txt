#===-- CMakeLists.txt ------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Build instructions for the flang-rt library. This is file is intended to be
# included using the LLVM_ENABLE_RUNTIMES mechanism.
#
#===------------------------------------------------------------------------===#

if (NOT LLVM_RUNTIMES_BUILD)
  message(FATAL_ERROR "Use this CMakeLists.txt from LLVM's runtimes build system.
      Example:
        cmake <llvm-project>/runtimes -DLLVM_ENABLE_RUNTIMES=flang-rt
    ")
endif ()

set(LLVM_SUBPROJECT_TITLE "Flang-RT")
set(FLANG_RT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FLANG_RT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(FLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../flang")

# CMake 3.24 is the first version of CMake that directly recognizes Flang.
# LLVM's requirement is only CMake 3.20, teach CMake 3.20-3.23 how to use Flang.
if (CMAKE_VERSION VERSION_LESS "3.24")
  cmake_path(GET CMAKE_Fortran_COMPILER STEM _Fortran_COMPILER_STEM)
  if (_Fortran_COMPILER_STEM STREQUAL "flang-new" OR _Fortran_COMPILER_STEM STREQUAL "flang")
    include(CMakeForceCompiler)
    CMAKE_FORCE_Fortran_COMPILER("${CMAKE_Fortran_COMPILER}" "LLVMFlang")

    set(CMAKE_Fortran_COMPILER_ID "LLVMFlang")
    set(CMAKE_Fortran_COMPILER_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")

    set(CMAKE_Fortran_SUBMODULE_SEP "-")
    set(CMAKE_Fortran_SUBMODULE_EXT ".mod")

    set(CMAKE_Fortran_PREPROCESS_SOURCE
      "<CMAKE_Fortran_COMPILER> -cpp <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")

    set(CMAKE_Fortran_FORMAT_FIXED_FLAG "-ffixed-form")
    set(CMAKE_Fortran_FORMAT_FREE_FLAG "-ffree-form")

    set(CMAKE_Fortran_MODDIR_FLAG "-module-dir")

    set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_ON "-cpp")
    set(CMAKE_Fortran_COMPILE_OPTIONS_PREPROCESS_OFF "-nocpp")
    set(CMAKE_Fortran_POSTPROCESS_FLAG "-ffixed-line-length-72")

    set(CMAKE_Fortran_COMPILE_OPTIONS_TARGET "--target=")

    set(CMAKE_Fortran_LINKER_WRAPPER_FLAG "-Wl,")
    set(CMAKE_Fortran_LINKER_WRAPPER_FLAG_SEP ",")
  endif ()
endif ()
enable_language(Fortran)


list(APPEND CMAKE_MODULE_PATH
    "${FLANG_RT_SOURCE_DIR}/cmake/modules"
    "${FLANG_SOURCE_DIR}/cmake/modules"
  )
include(AddFlangRT)
include(GetToolchainDirs)
include(FlangCommon)
include(HandleCompilerRT)
include(ExtendPath)


############################
# Build Mode Introspection #
############################

# Determine whether we are in the runtimes/runtimes-bins directory of a
# bootstrap build.
set(LLVM_TREE_AVAILABLE OFF)
if (LLVM_LIBRARY_OUTPUT_INTDIR AND LLVM_RUNTIME_OUTPUT_INTDIR AND PACKAGE_VERSION)
  set(LLVM_TREE_AVAILABLE ON)
endif()

# Path to LLVM development tools (FileCheck, llvm-lit, not, ...)
set(LLVM_TOOLS_DIR "${LLVM_BINARY_DIR}/bin")

# Determine build and install paths.
# The build path is absolute, but the install dir is relative, CMake's install
# command has to apply CMAKE_INSTALL_PREFIX itself.
get_toolchain_library_subdir(toolchain_lib_subdir)
if (LLVM_TREE_AVAILABLE)
  # In a bootstrap build emit the libraries into a default search path in the
  # build directory of the just-built compiler. This allows using the
  # just-built compiler without specifying paths to runtime libraries.
  #
  # Despite Clang in the name, get_clang_resource_dir does not depend on Clang
  # being added to the build. Flang uses the same resource dir as clang.
  include(GetClangResourceDir)
  get_clang_resource_dir(FLANG_RT_OUTPUT_RESOURCE_DIR PREFIX "${LLVM_LIBRARY_OUTPUT_INTDIR}/..")
  get_clang_resource_dir(FLANG_RT_INSTALL_RESOURCE_PATH_DEFAULT)

  extend_path(FLANG_RT_OUTPUT_RESOURCE_LIB_DIR "${FLANG_RT_OUTPUT_RESOURCE_DIR}" "${toolchain_lib_subdir}")
else ()
  # In a standalone runtimes build, do not write into LLVM_BINARY_DIR. It may be
  # read-only and/or shared by multiple runtimes with different build
  # configurations (e.g. Debug/Release). Use the runtime's own lib dir like any
  # non-toolchain library.
  # For the install prefix, still use the resource dir assuming that Flang will
  # be installed there using the same prefix. This is to not have a difference
  # between bootstrap and standalone runtimes builds.
  set(FLANG_RT_OUTPUT_RESOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(FLANG_RT_INSTALL_RESOURCE_PATH_DEFAULT "lib${LLVM_LIBDIR_SUFFIX}/clang/${LLVM_VERSION_MAJOR}")

  extend_path(FLANG_RT_OUTPUT_RESOURCE_LIB_DIR "${FLANG_RT_OUTPUT_RESOURCE_DIR}" "lib${LLVM_LIBDIR_SUFFIX}")
endif ()
set(FLANG_RT_INSTALL_RESOURCE_PATH "${FLANG_RT_INSTALL_RESOURCE_PATH_DEFAULT}"
    CACHE PATH "Path to install runtime libraries to (default: clang resource dir)")
extend_path(FLANG_RT_INSTALL_RESOURCE_LIB_PATH "${FLANG_RT_INSTALL_RESOURCE_PATH}" "${toolchain_lib_subdir}")
cmake_path(NORMAL_PATH FLANG_RT_OUTPUT_RESOURCE_DIR)
cmake_path(NORMAL_PATH FLANG_RT_INSTALL_RESOURCE_PATH)
# FIXME: For the libflang_rt.so, the toolchain resource lib dir is not a good
#        destination because it is not a ld.so default search path.
#        The machine where the executable is eventually executed may not be the
#        machine where the Flang compiler and its resource dir is installed, so
#        setting RPath by the driver is not an solution. It should belong into
#        /usr/lib/<triple>/libflang_rt.so, like e.g. libgcc_s.so.
#        But the linker as invoked by the Flang driver also requires
#        libflang_rt.so to be found when linking and the resource lib dir is
#        the only reliable location.
cmake_path(NORMAL_PATH FLANG_RT_OUTPUT_RESOURCE_LIB_DIR)
cmake_path(NORMAL_PATH FLANG_RT_INSTALL_RESOURCE_LIB_PATH)


#################
# Build Options #
#################

# Important: flang-rt user options must be prefixed with "FLANG_RT_". Variables
# with this prefix will be forwarded in bootstrap builds.

option(FLANG_RT_INCLUDE_TESTS "Generate build targets for the flang-rt unit and regression-tests." "${LLVM_INCLUDE_TESTS}")

# Provide an interface to link against the LLVM libc/libc++ projects directly.
set(FLANG_RT_SUPPORTED_PROVIDERS system llvm)
set(FLANG_RT_LIBC_PROVIDER "system" CACHE STRING "Specify C library to use. Supported values are ${FLANG_RT_SUPPORTED_PROVIDERS}.")
if (NOT "${FLANG_RT_LIBC_PROVIDER}" IN_LIST FLANG_RT_SUPPORTED_PROVIDERS)
  message(FATAL_ERROR "Unsupported library: '${FLANG_RT_RUNTIME_PROVIDER}'. Supported values are ${FLANG_RT_SUPPORTED_PROVIDERS}.")
endif ()

set(FLANG_RT_LIBCXX_PROVIDER "system" CACHE STRING "Specify C++ library to use. Supported values are ${FLANG_RT_SUPPORTED_PROVIDERS}.")
if (NOT "${FLANG_RT_LIBCXX_PROVIDER}" IN_LIST FLANG_RT_SUPPORTED_PROVIDERS)
  message(FATAL_ERROR "Unsupported library: '${FLANG_RT_LIBCXX_PROVIDER}'. Supported values are ${FLANG_RT_SUPPORTED_PROVIDERS}.")
endif ()

option(FLANG_RT_ENABLE_STATIC "Build Flang-RT as a static library." ON)
if (WIN32)
  # Windows DLL currently not implemented.
  set(FLANG_RT_ENABLE_SHARED OFF)
else ()
  # TODO: Enable by default to increase test coverage, and which version of the
  #       library should be the user's choice anyway.
  #       Currently, the Flang driver adds `-L"libdir" -lflang_rt` as linker
  #       argument, which leaves the choice which library to use to the linker.
  #       Since most linkers prefer the shared library, this would constitute a
  #       breaking change unless the driver is changed.
  option(FLANG_RT_ENABLE_SHARED "Build Flang-RT as a shared library." OFF)
endif ()
if (NOT FLANG_RT_ENABLE_STATIC AND NOT FLANG_RT_ENABLE_SHARED)
  message(FATAL_ERROR "
      Must build at least one type of library
      (FLANG_RT_ENABLE_STATIC=ON, FLANG_RT_ENABLE_SHARED=ON, or both)
    ")
endif ()


set(FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT "" CACHE STRING "Compile Flang-RT with GPU support (CUDA or OpenMP)")
set_property(CACHE FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT PROPERTY STRINGS
    ""
    CUDA
    OpenMP
  )
if (NOT FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT)
  # Support for GPUs disabled
elseif (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "CUDA")
  # Support for CUDA
  set(FLANG_RT_LIBCUDACXX_PATH "" CACHE PATH "Path to libcu++ package installation")
  option(FLANG_RT_CUDA_RUNTIME_PTX_WITHOUT_GLOBAL_VARS "Do not compile global variables' definitions when producing PTX library" OFF)
elseif (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "OpenMP")
  # Support for OpenMP offloading
  set(FLANG_RT_DEVICE_ARCHITECTURES "all" CACHE STRING
      "List of OpenMP device architectures to be used to compile the Fortran runtime (e.g. 'gfx1103;sm_90')"
    )

  if (FLANG_RT_DEVICE_ARCHITECTURES STREQUAL "all")
    # TODO: support auto detection on the build system.
    set(all_amdgpu_architectures
      "gfx700;gfx701;gfx801;gfx803;gfx900;gfx902;gfx906"
      "gfx908;gfx90a;gfx90c;gfx940;gfx1010;gfx1030"
      "gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036"
      "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151"
      "gfx1152;gfx1153")
    set(all_nvptx_architectures
      "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62"
      "sm_70;sm_72;sm_75;sm_80;sm_86;sm_89;sm_90")
    set(all_gpu_architectures
      "${all_amdgpu_architectures};${all_nvptx_architectures}")
      set(FLANG_RT_DEVICE_ARCHITECTURES ${all_gpu_architectures})
  endif()
  list(REMOVE_DUPLICATES FLANG_RT_DEVICE_ARCHITECTURES)
else ()
  message(FATAL_ERROR "Invalid value '${FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT}' for FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT; must be empty, 'CUDA', or 'OpenMP'")
endif ()


option(FLANG_RT_INCLUDE_CUF "Build the CUDA Fortran runtime (libflang_rt.cuda.a)" OFF)
if (FLANG_RT_INCLUDE_CUF)
  find_package(CUDAToolkit REQUIRED)
endif()


########################
# System Introspection #
########################

# The GPU targets require a few mandatory arguments to make the standard CMake
# check flags happy.
if ("${LLVM_RUNTIMES_TARGET}" MATCHES "^amdgcn")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -nogpulib")
elseif ("${LLVM_RUNTIMES_TARGET}" MATCHES "^nvptx")
  set(CMAKE_REQUIRED_FLAGS
      "${CMAKE_REQUIRED_FLAGS} -flto -c -Wno-unused-command-line-argument")
endif()

include(CheckCXXSymbolExists)
include(CheckCXXSourceCompiles)
check_cxx_symbol_exists(strerror_r string.h HAVE_STRERROR_R)
# Can't use symbol exists here as the function is overloaded in C++
check_cxx_source_compiles(
  "#include <string.h>
   int main() {
     char buf[4096];
     return strerror_s(buf, 4096, 0);
   }
  "
  HAVE_DECL_STRERROR_S)

# Search for clang_rt.builtins library. Need in addition to msvcrt.
if (WIN32)
  find_compiler_rt_library(builtins FLANG_RT_BUILTINS_LIBRARY)
endif ()

# Build with _XOPEN_SOURCE on AIX to avoid errors caused by _ALL_SOURCE.
# We need to enable the large-file API as well.
if (UNIX AND CMAKE_SYSTEM_NAME MATCHES "AIX")
  add_compile_definitions(_XOPEN_SOURCE=700)
  add_compile_definitions(_LARGE_FILE_API)
endif ()

# Check whether the compiler can undefine a macro using the "-Wp,-U" flag.
# Aternatively, we could use
#   CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
# but some older versions of CMake don't define it for GCC itself.
check_cxx_compiler_flag("-Wp,-UTESTFLAG" FLANG_RT_SUPPORTS_UNDEFINE_FLAG)

# Check whether -fno-lto is supported.
check_cxx_compiler_flag(-fno-lto FLANG_RT_HAS_FNO_LTO_FLAG)

# Check whether -nostdlibinc is supported.
check_cxx_compiler_flag(-nostdlibinc FLANG_RT_HAS_NOSTDLIBINC_FLAG)

# Check whether -nostdlib is supported.
check_cxx_compiler_flag(-nostdlib FLANG_RT_HAS_NOSTDLIB_FLAG)

# Check whether -stdlib= is supported.
check_cxx_compiler_flag(-stdlib=platform FLANG_RT_HAS_STDLIB_FLAG)

# Check whether -Wl,--as-needed is supported.
check_linker_flag(C "LINKER:--as-needed" LINKER_SUPPORTS_AS_NEEDED)
if (LINKER_SUPPORTS_AS_NEEDED)
  set(LINKER_AS_NEEDED_OPT "LINKER:--as-needed")
endif()

# Different platform may have different name for the POSIX thread library.
# For example, libpthread.a on AIX. Search for it as it is needed when
# building the shared flang_rt.runtime.so.
find_package(Threads)

# function checks
find_package(Backtrace)
set(HAVE_BACKTRACE ${Backtrace_FOUND})
set(BACKTRACE_HEADER ${Backtrace_HEADER})


#####################
# Build Preparation #
#####################

include(HandleLibs)

if (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT AND FLANG_RT_INCLUDE_TESTS)
  # If Fortran runtime is built as CUDA library, the linking
  # of targets that link flang-rt must be done
  # with CUDA_RESOLVE_DEVICE_SYMBOLS.
  # CUDA language must be enabled for CUDA_RESOLVE_DEVICE_SYMBOLS
  # to take effect.
  enable_language(CUDA)
endif()


# C++17 is required for flang-rt; user or other runtimes may override this.
# GTest included later also requires C++17.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)


configure_file(cmake/config.h.cmake.in config.h)
if (FLANG_INCLUDE_QUADMATH_H)
  configure_file("cmake/quadmath_wrapper.h.in" "${FLANG_RT_BINARY_DIR}/quadmath_wrapper.h")
endif ()

# The bootstrap build will create a phony target with the same as the top-level
# directory ("flang-rt") and delegate it to the runtimes build dir.
# AddFlangRT will add all non-EXCLUDE_FROM_ALL targets to it.
add_custom_target(flang-rt)


###################
# Build Artifacts #
###################

add_subdirectory(lib)

if (LLVM_INCLUDE_EXAMPLES)
  add_subdirectory(examples)
endif ()

if (FLANG_RT_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
else ()
  add_custom_target(check-flang-rt)
endif()
