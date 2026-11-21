#===-- cmake/modules/FlangCommon.txt ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# CMake definitions shared between Flang and Flang-RT
#
#===------------------------------------------------------------------------===#

include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckIncludeFile)
include(CMakePushCheckState)

# Check whether the compiler honors '#pragma STDC FENV_ACCESS ON'
set(FLANG_SAVED_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
cmake_push_check_state()
if(MSVC)
  string(APPEND CMAKE_REQUIRED_FLAGS " /WX")
else()
  string(APPEND CMAKE_REQUIRED_FLAGS " -Wall -Werror")
endif()
check_cxx_source_compiles(
  "int probe() {
   #pragma STDC FENV_ACCESS ON
     return 0;
   }
  "
  COMPILER_SUPPORTS_STDC_FENV_ACCESS_PRAGMA)
cmake_pop_check_state()
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${FLANG_SAVED_TRY_COMPILE_TARGET_TYPE}")
unset(FLANG_SAVED_TRY_COMPILE_TARGET_TYPE)

if(COMPILER_SUPPORTS_STDC_FENV_ACCESS_PRAGMA)
  add_compile_definitions(HAVE_STDC_FENV_ACCESS)
endif()

# The out of tree builds of the compiler and the Fortran runtime
# must use the same setting of FLANG_RUNTIME_F128_MATH_LIB
# to be composable. Failure to synchronize this setting may result
# in linking errors or fatal failures in F128 runtime functions.
set(FLANG_RUNTIME_F128_MATH_LIB "" CACHE STRING
  "Specifies the target library used for implementing IEEE-754 128-bit float \
  math in F18 runtime, e.g. it might be libquadmath for targets where \
  REAL(16) is mapped to __float128, or libm for targets where REAL(16) \
  is mapped to long double, etc."
  )
if (FLANG_RUNTIME_F128_MATH_LIB)
  add_compile_definitions(FLANG_RUNTIME_F128_MATH_LIB="${FLANG_RUNTIME_F128_MATH_LIB}")
endif()

# Check if 128-bit float computations can be done via long double
# Note that '-nostdinc++' might be implied when this code kicks in
# (see 'runtimes/CMakeLists.txt'), so we cannot use 'cfloat' C++ header
# file in the test below.
# Compile it as C.
check_c_source_compiles(
  "#include <float.h>
   #if LDBL_MANT_DIG != 113
   #error LDBL_MANT_DIG != 113
   #endif
   int main() { return 0; }
  "
  HAVE_LDBL_MANT_DIG_113)

# Discover the GCC installation, when the build compiler is Clang,
# and try to find quadmath.h there. Set FLANG_INCLUDE_QUADMATH_H
# to the path to quadmath.h, if found.
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (NOT DEFINED FLANG_GCC_RESOURCE_DIR)
    set(FLANG_GCC_RESOURCE_DIR "FLANG_GCC_RESOURCE_DIR-NOTFOUND")
    # Prepare CMAKE_CXX_FLAGS so that they can be passed to execute_process
    # as separate flags.
    separate_arguments(flags UNIX_COMMAND "${CMAKE_CXX_FLAGS}")
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" ${flags} -v "-###"
      ERROR_FILE "${CMAKE_CURRENT_BINARY_DIR}/clang_gcc_root_result"
    )
    file(STRINGS "${CMAKE_CURRENT_BINARY_DIR}/clang_gcc_root_result" _errorresult)
    foreach (_line IN LISTS _errorresult)
      string(REGEX MATCH
        "^Selected GCC installation: (.+)$"
        _match
        "${_line}")
      if (CMAKE_MATCH_1)
        set(FLANG_GCC_RESOURCE_DIR "${CMAKE_MATCH_1}")
        message(STATUS "Found GCC installation selected by Clang: ${FLANG_GCC_RESOURCE_DIR}")
        break()
      endif ()
    endforeach ()
    set(FLANG_GCC_RESOURCE_DIR "${FLANG_GCC_RESOURCE_DIR}" CACHE INTERNAL "Path to GCC's resource dir selected by Clang" FORCE)
  endif ()
endif ()

check_include_file("quadmath.h" FOUND_QUADMATH_H)
if (FOUND_QUADMATH_H)
  message(STATUS "quadmath.h found without additional include paths")
  set(FLANG_INCLUDE_QUADMATH_H "<quadmath.h>")
elseif (FLANG_GCC_RESOURCE_DIR)
  cmake_push_check_state()
    list(APPEND CMAKE_REQUIRED_INCLUDES "${FLANG_GCC_RESOURCE_DIR}/include")
    check_include_file("quadmath.h" FOUND_GCC_QUADMATH_H)
  cmake_pop_check_state()
  if (FOUND_GCC_QUADMATH_H)
    message(STATUS "quadmath.h found in Clang's selected GCC installation")
    set(FLANG_INCLUDE_QUADMATH_H "\"${FLANG_GCC_RESOURCE_DIR}/include/quadmath.h\"")
  endif ()
endif ()
