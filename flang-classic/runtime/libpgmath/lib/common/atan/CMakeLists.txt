
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L1")
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L1")


set(SRCS_SCALAR
  fd_atan_scalar.cpp
  )

set(SRCS_VECTOR
  fd_atan_vector.cpp
  )

list(APPEND DEFINITIONS NDEBUG)
if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND DEFINITIONS _GNU_SOURCE)
endif()

# Scalar 
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_CPU=avx2")
libmath_add_object_library("${SRCS_SCALAR}" "${FLAGS_TMP}" "${DEFINITIONS}"
"atan-avx2_1")


# Vector, Two elements
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_CPU=avx2 -D_VL=2")
libmath_add_object_library("${SRCS_VECTOR}" "${FLAGS_TMP}" "${DEFINITIONS}"
"atan-avx2_2")


# Vector, Four elements
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_CPU=avx2 -D_VL=4")
libmath_add_object_library("${SRCS_VECTOR}" "${FLAGS_TMP}" "${DEFINITIONS}"
"atan-avx2_4")


# Vector, Eight elements
set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_CPU=avx512 -D_VL=8")
libmath_add_object_library("${SRCS_VECTOR}" "${FLAGS_TMP}" "${DEFINITIONS}"
"atan-avx512_8")

