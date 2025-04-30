#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler definitions and flags
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L2")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L2")

# Set source files
set(SRCS
  vsatan4.cpp
  vsatan8.cpp
  vsatan_mask.S
  ssatan.cpp)
libmath_add_object_library("${SRCS}" "${FLAGS} -mfma -mavx2" "${DEFINITIONS}" "")
set(SRCS
  vsatan16.cpp)
set(FLAGS_AVX512 "${FLAGS} -march=skylake-avx512 -mtune=skylake-avx512")
libmath_add_object_library("${SRCS}" "${FLAGS_AVX512}" "${DEFINITIONS}" "atan-avx512")
set(FLAGS_AVX512 "${FLAGS} -march=knl -mtune=knl")
libmath_add_object_library("${SRCS}" "${FLAGS_AVX512}" "${DEFINITIONS}" "atan-knl")
