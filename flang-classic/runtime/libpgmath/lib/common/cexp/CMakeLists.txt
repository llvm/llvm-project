#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L1")
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L1")


set(SRCS
  cexp.c
  )

list(APPEND DEFINITIONS NDEBUG)
if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND DEFINITIONS _GNU_SOURCE)
endif()

# Scalar 
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_VL=1 -D_CPU=avx2")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx2_1")

set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_VL=1 -D_CPU=avx512")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx512_1")

set(FLAGS_TMP "${FLAGS} -mtune=knl -march=knl -D_VL=1 -D_CPU=knl")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-knl_1")


# Scalar with vector calling ABI
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_VL=1 -D_CPU=avx2 -D_SCALAR_WITH_VECTOR_ABI_")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx2_1v")

set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_VL=1 -D_CPU=avx512 -D_SCALAR_WITH_VECTOR_ABI_")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx512_1v")

set(FLAGS_TMP "${FLAGS} -mtune=knl -march=knl -D_VL=1 -D_CPU=knl -D_SCALAR_WITH_VECTOR_ABI_")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-knl_1v")


# Vector, Two elements
set(FLAGS_TMP "${FLAGS} -mtune=core-avx2 -march=core-avx2 -D_VL=2 -D_CPU=avx2")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx2_2")

set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_VL=2 -D_CPU=avx512")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx512_2")

set(FLAGS_TMP "${FLAGS} -mtune=knl -march=knl -D_VL=2 -D_CPU=knl")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-knl_2")


# Vector, Four elements
set(FLAGS_TMP "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512 -D_VL=4 -D_CPU=avx512")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-avx512_4")

set(FLAGS_TMP "${FLAGS} -mtune=knl -march=knl -D_VL=4 -D_CPU=knl")
libmath_add_object_library("${SRCS}" "${FLAGS_TMP}" "${DEFINITIONS}" "cexp-knl_4")

