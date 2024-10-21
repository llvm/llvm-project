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

set(LLVM_SUBPROJECT_TITLE "Fortran Runtime")
set(FLANGRT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FLANGRT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(FLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../flang")

enable_language(Fortran)

list(APPEND CMAKE_MODULE_PATH
    "${FLANGRT_SOURCE_DIR}/cmake/modules"
    "${FLANG_SOURCE_DIR}/cmake/modules"
  )
include(AddFlangRT)
include(FlangCommon)


############################
# Build Mode Introspection #
############################

# Setting these variables from an LLVM build is sufficient that flang-rt can
# construct the output paths, so it can behave as if it were in-tree here.
set(LLVM_TREE_AVAILABLE OFF)
if (LLVM_LIBRARY_OUTPUT_INTDIR AND LLVM_RUNTIME_OUTPUT_INTDIR AND PACKAGE_VERSION)
  # This is a bootstap build
  set(LLVM_TREE_AVAILABLE ON)
endif()

if (LLVM_TREE_AVAILABLE)
  # Despite Clang in the name, get_clang_resource_dir does not depend on Clang being added to the build
  # flang-new uses the same resource dir as clang.
  include(GetClangResourceDir)
  get_clang_resource_dir(FLANGRT_BUILD_LIB_DIR PREFIX "${LLVM_LIBRARY_OUTPUT_INTDIR}/.." SUBDIR "lib${LLVM_LIBDIR_SUFFIX}")
  get_clang_resource_dir(FLANGRT_INSTALL_LIB_DIR SUBDIR "lib${LLVM_LIBDIR_SUFFIX}") # No prefix, CMake's install command finds the install prefix itself
else ()
  set(FLANGRT_BUILD_LIB_DIR "${LLVM_LIBRARY_OUTPUT_INTDIR}")
  set(FLANGRT_INSTALL_LIB_DIR "lib${LLVM_LIBDIR_SUFFIX}")
endif ()

if (DEFINED WIN32)
  set(FLANGRT_BUILD_LIB_DIR "${FLANGRT_BUILD_LIB_DIR}/windows")
  set(FLANGRT_INSTALL_LIB_DIR "${FLANGRT_INSTALL_LIB_DIR}/windows")
elseif (LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  set(FLANGRT_BUILD_LIB_DIR "${FLANGRT_BUILD_LIB_DIR}/${LLVM_TARGET_TRIPLE}")
  set(FLANGRT_INSTALL_LIB_DIR "${FLANGRT_INSTALL_LIB_DIR}/${LLVM_TARGET_TRIPLE}")
endif ()


#################
# Build Options #
#################

# Important: flang-rt user options must be prefixed with "FLANG_RT_". Variables
# with this prefix will be forwarded in bootstrap builds.

option(FLANG_RT_INCLUDE_TESTS "Generate build targets for the flang-rt unit and regression-tests." "${LLVM_INCLUDE_TESTS}")

set(FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT "" CACHE STRING "Compile flang-rt with GPU support (CUDA or OpenMP)")
set_property(CACHE FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT PROPERTY STRINGS
    ""
    CUDA
    OpenMP
  )
if (NOT FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT)
elseif (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "CUDA")
  set(FLANG_RT_LIBCUDACXX_PATH "" CACHE PATH "Path to libcu++ package installation")
  option(FLANG_RT_CUDA_RUNTIME_PTX_WITHOUT_GLOBAL_VARS "Do not compile global variables' definitions when producing PTX library" OFF)
elseif (FLANG_RT_EXPERIMENTAL_OFFLOAD_SUPPORT STREQUAL "OpenMP")
set(FLANG_RT_DEVICE_ARCHITECTURES "all" CACHE STRING
  "List of OpenMP device architectures to be used to compile the Fortran runtime (e.g. 'gfx1103;sm_90')")
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


# Search for clang_rt.builtins library.
if (WIN32)
  execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" "-print-libgcc-file-name" "-rtlib=compiler-rt"
      RESULT_VARIABLE CXX_COMPILER_PRINT_LIBGCC_PATH_FAILURE
      OUTPUT_VARIABLE CXX_COMPILER_PRINT_LIBGCC_PATH_RESULT
      ERROR_QUIET
    )
  if (NOT CXX_COMPILER_PRINT_LIBGCC_PATH_FAILURE AND CXX_COMPILER_PRINT_LIBGCC_PATH_RESULT)
    string(STRIP "${CXX_COMPILER_PRINT_LIBGCC_PATH_RESULT}" FLANGRT_LIBCALL)
  else ()
    set(FLANGRT_LIBCALL "")
  endif ()
endif ()


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
