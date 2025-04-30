
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L1")
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L1")


set(SRCS_SCALAR
  fs_log10_scalar.cpp
  )

set(SRCS_SCALAR_LEGACY
  fs_log10_scalar_legacy.cpp
  )

set(SRCS_VECTOR4
  fs_log10_4.cpp
  )

set(SRCS_VECTOR8
  fs_log10_8.cpp
  )

set(SRCS_VECTOR16
  fs_log10_16.cpp
  )

set(SRCS_VECTOR_LEGACY
  fs_log10_vector_legacy.cpp
  )

list(APPEND DEFINITIONS NDEBUG)
if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND DEFINITIONS _GNU_SOURCE)
endif()

# Scalar 
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_CPU=avx2")
libmath_add_object_library("${SRCS_SCALAR_LEGACY}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx2_1")

set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_CPU=avx512")
libmath_add_object_library("${SRCS_SCALAR}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx512_1")


# Vector, Four elements
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_CPU=avx2 -D_VL=4")
libmath_add_object_library("${SRCS_VECTOR_LEGACY}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx2_4")

set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_CPU=avx512 -D__AVX512VL__")
libmath_add_object_library("${SRCS_VECTOR4}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx512_4")

set(FLAGS_TMP "${FLAGS} -mtune=knl -march=knl -D_CPU=knl")
libmath_add_object_library("${SRCS_VECTOR4}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-knl_4")


# Vector, Eight elements
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_CPU=avx2 -D_VL=8")
libmath_add_object_library("${SRCS_VECTOR_LEGACY}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx2_8")

set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_CPU=avx512 -D__AVX512VL__")
libmath_add_object_library("${SRCS_VECTOR8}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx512_8")

set(FLAGS_TMP "${FLAGS} -mtune=knl -march=knl -D_CPU=knl")
libmath_add_object_library("${SRCS_VECTOR8}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-knl_8")


# Vector, Sixteen elements
set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_CPU=avx512 -D__AVX512F__")
libmath_add_object_library("${SRCS_VECTOR16}" "${FLAGS_TMP}" "${DEFINITIONS}" "log10f-avx512_16")

