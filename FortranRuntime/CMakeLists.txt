#===-- CMakeLists.txt ------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Build instructions for the FortranRuntime library and Fortran modules required
# by programs compiled by Flang. This is file is intended to be included using
# the LLVM_ENABLE_RUNTIMES mechanism.
#
#===------------------------------------------------------------------------===#

set(LLVM_SUBPROJECT_TITLE "Fortran Runtime")
set(FORTRANRUNTIME_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FORTRANRUNTIME_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(FLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../flang")

enable_language(Fortran)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )
include(AddFortranRuntime)
include(FlangCommon)


############################
# Build Mode Introspection #
############################

# Setting these variables from an LLVM build is sufficient that FortranRuntime can
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
  get_clang_resource_dir(FORTRANRUNTIME_BUILD_LIB_DIR PREFIX "${LLVM_LIBRARY_OUTPUT_INTDIR}/.." SUBDIR "lib${LLVM_LIBDIR_SUFFIX}")
  get_clang_resource_dir(FORTRANRUNTIME_INSTALL_LIB_DIR SUBDIR "lib${LLVM_LIBDIR_SUFFIX}") # No prefix, CMake's install command find the install prefix itself
else ()
  set(FORTRANRUNTIME_BUILD_LIB_DIR "${LLVM_LIBRARY_OUTPUT_INTDIR}")
  set(FORTRANRUNTIME_INSTALL_LIB_DIR "lib${LLVM_LIBDIR_SUFFIX}")
endif ()

if (DEFINED WIN32)
  set(FORTRANRUNTIME_BUILD_LIB_DIR "${FORTRANRUNTIME_BUILD_LIB_DIR}/windows")
  set(FORTRANRUNTIME_INSTALL_LIB_DIR "${FORTRANRUNTIME_INSTALL_LIB_DIR}/windows")
elseif (LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  set(FORTRANRUNTIME_BUILD_LIB_DIR "${FORTRANRUNTIME_BUILD_LIB_DIR}/${LLVM_TARGET_TRIPLE}")
  set(FORTRANRUNTIME_INSTALL_LIB_DIR "${FORTRANRUNTIME_INSTALL_LIB_DIR}/${LLVM_TARGET_TRIPLE}")
endif ()


#################
# Build Options #
#################

option(FORTRANRUNTIME_INCLUDE_TESTS
         "Generate build targets for the FortranRuntime unit tests."
         ${LLVM_INCLUDE_TESTS})

option(FLANG_EXPERIMENTAL_CUDA_RUNTIME "Compile Fortran runtime as CUDA sources (experimental)" OFF)
if (FLANG_EXPERIMENTAL_CUDA_RUNTIME)
  message(FATAL_ERROOR "FLANG_EXPERIMENTAL_CUDA_RUNTIME currently not supported")
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


# Search for clang_rt.builtins
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" "-print-libgcc-file-name" "-rtlib=compiler-rt"
    RESULT_VARIABLE CXX_COMPILER_PRINT_LIBGCC_PATH_FAILURE
    OUTPUT_VARIABLE CXX_COMPILER_PRINT_LIBGCC_PATH_RESULT
    ERROR_QUIET
  )
if (NOT CXX_COMPILER_PRINT_LIBGCC_PATH_FAILURE AND CXX_COMPILER_PRINT_LIBGCC_PATH_RESULT)
  string(STRIP "${CXX_COMPILER_PRINT_LIBGCC_PATH_RESULT}" FORTRANRUNTIME_LIBCALL)
else ()
  set(FORTRANRUNTIME_LIBCALL "")
endif ()


#####################
# Build Preparation #
#####################

# C++17 required for FortranRuntime, user or other runtimes may override this.
# GTest included later also requires C++17.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Disable libstdc++/libc++ assertions, even in an LLVM_ENABLE_ASSERTIONS build,
# to avoid an unwanted dependency on libstdc++/libc++.so.
add_definitions(-U_GLIBCXX_ASSERTIONS)
add_definitions(-U_LIBCPP_ENABLE_ASSERTIONS)

configure_file(cmake/config.h.cmake.in config.h)


###################
# Build Artifacts #
###################

add_subdirectory(lib)

if (FORTRANRUNTIME_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()
