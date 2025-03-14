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

include(TestBigEndian)
test_big_endian(IS_BIGENDIAN)
if (IS_BIGENDIAN)
  add_compile_definitions(FLANG_BIG_ENDIAN=1)
else ()
  add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
endif ()
