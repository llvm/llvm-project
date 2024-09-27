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

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )
include(AddFortranRuntime)



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

  # FIXME: If compiling multiple triples, include dirs overwrite each other
  get_clang_resource_dir(FORTRANRUNTIME_BUILD_INCLUDE_DIR PREFIX "${LLVM_LIBRARY_OUTPUT_INTDIR}/.." SUBDIR "include")
  get_clang_resource_dir(FORTRANRUNTIME_INSTALL_INCLUDE_DIR SUBDIR "include") 
else ()
  set(FORTRANRUNTIME_BUILD_LIB_DIR "${LLVM_LIBRARY_OUTPUT_INTDIR}")
  set(FORTRANRUNTIME_INSTALL_LIB_DIR "lib${LLVM_LIBDIR_SUFFIX}")

  set(FORTRANRUNTIME_BUILD_INCLUDE_DIR "${LLVM_INCLUDE_DIR}")
  set(FORTRANRUNTIME_INSTALL_INCLUDE_DIR "include")
endif ()


if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  set(FORTRANRUNTIME_BUILD_LIB_DIR "${FORTRANRUNTIME_BUILD_LIB_DIR}/${LLVM_TARGET_TRIPLE}")
  set(FORTRANRUNTIME_INSTALL_LIB_DIR "${FORTRANRUNTIME_INSTALL_LIB_DIR}/${LLVM_TARGET_TRIPLE}")
endif()




#################
# Build Options #
#################

option(FORTRANRUNTIME_INCLUDE_TESTS
         "Generate build targets for the FortranRuntime unit tests."
         ${LLVM_INCLUDE_TESTS})

# The out of tree builds of the compiler and the Fortran runtime
# must use the same setting of FLANG_RUNTIME_F128_MATH_LIB
# to be composable. Failure to synchronize this setting may result
# in linking errors or fatal failures in F128 runtime functions.
set(FORTRARUNTIME_F128_MATH_LIB "" CACHE STRING
  "Specifies the target library used for implementing IEEE-754 128-bit float \
  math in F18 runtime, e.g. it might be libquadmath for targets where \
  REAL(16) is mapped to __float128, or libm for targets where REAL(16) \
  is mapped to long double, etc."
  )
if (FLANG_RUNTIME_F128_MATH_LIB)
  add_compile_definitions(
    FLANG_RUNTIME_F128_MATH_LIB="${FLANG_RUNTIME_F128_MATH_LIB}"
    )
endif()

option(FLANG_EXPERIMENTAL_CUDA_RUNTIME "Compile Fortran runtime as CUDA sources (experimental)" OFF)
if (FLANG_EXPERIMENTAL_CUDA_RUNTIME)
  message(FATAL_ERROOR "FLANG_EXPERIMENTAL_CUDA_RUNTIME currently not supported")
endif()




########################
# System Introspection #
########################

include(TestBigEndian)
test_big_endian(IS_BIGENDIAN)
if (IS_BIGENDIAN)
  add_compile_definitions(FLANG_BIG_ENDIAN=1)
else ()
  add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
endif ()

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

# Check if 128-bit float computations can be done via long double
check_cxx_source_compiles(
  "#include <cfloat>
   #if LDBL_MANT_DIG != 113
   #error LDBL_MANT_DIG != 113
   #endif
   int main() { return 0; }
  "
  HAVE_LDBL_MANT_DIG_113)

# Check if 128-bit float computations can be done via long double
check_cxx_source_compiles(
  "#include <cfloat>
   #if LDBL_MANT_DIG != 113
   #error LDBL_MANT_DIG != 113
   #endif
   int main() { return 0; }
  "
  HAVE_LDBL_MANT_DIG_113)


#####################
# Build Preparation #
#####################

enable_language(Fortran)

# C++17 required for FortranRuntime, user or other runtimes may override this.
# GTest included later also requires C++17.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to") 
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Disable libstdc++/libc++ assertions, even in an LLVM_ENABLE_ASSERTIONS build,
# to avoid an unwanted dependency on libstdc++/libc++.so.
add_definitions(-U_GLIBCXX_ASSERTIONS)
add_definitions(-U_LIBCPP_ENABLE_ASSERTIONS)


configure_file(cmake/config.h.cmake.in config.h)

configure_file(
  "${FORTRANRUNTIME_SOURCE_DIR}/include/flang/Common/ISO_Fortran_binding.h"
  "${FORTRANRUNTIME_BUILD_INCLUDE_DIR}/ISO_Fortran_binding.h"
)
install(
  FILES   "${FORTRANRUNTIME_BUILD_INCLUDE_DIR}/ISO_Fortran_binding.h"
  DESTINATION "${FORTRANRUNTIME_BUILD_INCLUDE_DIR}"
)



###################
# Build Artifacts #
###################

add_subdirectory(lib)
add_subdirectory(module)

if (FORTRANRUNTIME_INCLUDE_TESTS)
  add_subdirectory(unittests)
  add_subdirectory(test)
endif()

