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
include(GNUInstallDirs)


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
  get_clang_resource_dir(FLANG_RT_OUTPUT_DIR PREFIX "${LLVM_LIBRARY_OUTPUT_INTDIR}/..")
  get_clang_resource_dir(FLANG_RT_INSTALL_PATH)

  extend_path(FLANG_RT_BUILD_TOOLCHAIN_LIB_DIR "${FLANG_RT_OUTPUT_DIR}" "${toolchain_lib_subdir}")
else ()
  # In a standalone runtimes build, do not write into LLVM_BINARY_DIR. It may be
  # read-only and/or shared by multiple runtimes with different build
  # configurations (e.g. Debug/Release). Use the runtime's own lib dir like any
  # non-toolchain library.
  # For the install prefix, still use the resource dir assuming that Flang will
  # be installed there using the same prefix. This is to not have a difference
 # between bootstrap and standalone runtimes builds.
  set(FLANG_RT_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(FLANG_RT_INSTALL_PATH "${CMAKE_INSTALL_LIBDIR}/clang/${LLVM_VERSION_MAJOR}")

  extend_path(FLANG_RT_BUILD_TOOLCHAIN_LIB_DIR "${FLANG_RT_OUTPUT_DIR}" "${CMAKE_INSTALL_LIBDIR}")
endif ()

# Apply the arch-specific library dir with the resource dir.
extend_path(FLANG_RT_INSTALL_TOOLCHAIN_LIB_DIR "${FLANG_RT_INSTALL_PATH}" "${toolchain_lib_subdir}")

cmake_path(NORMAL_PATH FLANG_RT_OUTPUT_DIR)
cmake_path(NORMAL_PATH FLANG_RT_INSTALL_PATH)
cmake_path(NORMAL_PATH FLANG_RT_BUILD_TOOLCHAIN_LIB_DIR)
cmake_path(NORMAL_PATH FLANG_RT_INSTALL_TOOLCHAIN_LIB_DIR)


#################
# Build Options #
#################

# Important: flang-rt user options must be prefixed with "FLANG_RT_". Variables
# with this prefix will be forwarded in bootstrap builds.

option(FLANG_RT_INCLUDE_TESTS "Generate build targets for the flang-rt unit and regression-tests." "${LLVM_INCLUDE_TESTS}")


option(FLANG_RT_ENABLE_STATIC "Build Flang-RT as a static library." ON)
if (WIN32)
  # Windows DLL currently not implemented.
  set(FLANG_RT_ENABLE_SHARED OFF)
else ()
  option(FLANG_RT_ENABLE_SHARED "Build Flang-RT as a shared library." ON)
endif ()
if (NOT FLANG_RT_ENABLE_STATIC AND NOT FLANG_RT_ENABLE_SHARED)
  message(FATAL_ERROR "Must build at least one type of library
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
else ()
  message(FATAL_ERROR "Invalid value '${FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT}' for FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT; must be empty, 'CUDA', or 'OpenMP'")
endif ()


option(FLANG_RT_ENABLE_CUF "Compile CUDA Fortran runtime sources" OFF)
if (FLANG_RT_ENABLE_CUF)
  find_package(CUDAToolkit REQUIRED)
endif()


########################
# System Introspection #
########################

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


# Check whether the compiler can undefine a macro using the "-U" flag.
# Aternatively, we could use
#   CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
# but some older versions of CMake don't define it for GCC itself.
check_cxx_compiler_flag("-UTESTFLAG" FLANG_RT_SUPPORTS_UNDEFINE_FLAG)


#####################
# Build Preparation #
#####################

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


# The bootstrap build will create a phony target with the same as the top-level
# directory ("flang-rt") and delegate it to the runtimes build dir.
# AddFlangRT will add all non-EXCLUDE_FROM_ALL targets to it.
add_custom_target(flang-rt)


###################
# Build Artifacts #
###################

add_subdirectory(lib)

if (FLANG_RT_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
else ()
  add_custom_target(check-flang-rt)
endif()
