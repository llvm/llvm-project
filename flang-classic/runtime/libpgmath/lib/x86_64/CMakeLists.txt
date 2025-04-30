#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include(CheckCCompilerFlag)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory("acos")
add_subdirectory("asin")
add_subdirectory("atan")
add_subdirectory("exp")
add_subdirectory("fast")
add_subdirectory("log")
add_subdirectory("math_tables")
add_subdirectory("pow")
add_subdirectory("powi")
add_subdirectory("relaxed")

get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L1")
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L1")

set(ASM_SRCS
  aint.S
  anint.S
  around.S
  dint.S
  dnint.S
  dpow.S
  dremainder.S
  dround.S
  dsincos.S
  remainder.S
  sincos.S)

foreach (SRC_FILE anint.S dnint.S)
    set_source_files_properties(${SRC_FILE} PROPERTIES COMPILE_FLAGS -mavx512f)
endforeach()

set(SRCS
  abs.c
  acos.c
  acosf.c
  acosh.c
  asin.c
  asinf.c
  asinh.c
  atanh.c
  dacosh.c
  dasinh.c
  datanh.c
  dsqrt.c
  fabs.c
  sqrt.c
  pgcpuid.c
  x86id.c
  ${ASM_SRCS})
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")

# isoc99
set(SRCS
  alog.c
  atan.c
  atan2.c
  cosh.c
  datan.c
  datan2.c
  dcosh.c
  dexp.c
  dexp2.c
  dlog.c
  dsincosp.c
  dsinh.c
  dtan.c
  dtanh.c
  exp.c
  log.c
  rpow.c
  sincosp.c
  sinh.c
  tan.c
  tanh.c)
list(APPEND DEFINITIONS _GNU_SOURCE _ISOC99_SOURCE)
if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND DEFINITIONS _GNU_SOURCE)
endif()
set(FLAGS "${FLAGS}-Wall -W -Wstrict-prototypes -Wwrite-strings ")
if(NOT ${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
  set(FLAGS "${FLAGS}-Werror ")
endif()
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "isoc99")

# isoc99_log2
set(SRCS
  dlog.c
  alog.c
  log.c)
list(APPEND DEFINITIONS_LOG2 ${DEFINITIONS} COMPILING_LOG2)
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS_LOG2}" "isoc99_log2")

# isoc99_log10
set(SRCS
  dlog.c
  alog.c
  log.c)
list(APPEND DEFINITIONS_LOG10 ${DEFINITIONS} COMPILING_LOG10)
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS_LOG10}" "isoc99_log10")

# AVX2 + FMA3
set(SRCS
  alog.c
  atan2.c
  atan.c
  cosh.c
  datan2.c
  datan.c
  dcosh.c
  dexp2.c
  dexp.c
  dlog.c
  dsincosp.c
  dsinh.c
  dtan.c
  dtanh.c
  exp.c
  log.c
  rpow.c
  sincosp.c
  sinh.c
  tan.c
  tanh.c)
set(FLAGS "${FLAGS} -mavx2 -mfma")
if(NOT ${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
  set(FLAGS "${FLAGS} -ffp-contract=fast")
elseif (${CMAKE_COMPILER_ID} MATCHES "Clang")
  set(FLAGS "${FLAGS} -Xclang -ffp-contract=fast -Xclang -Wno-language-extension-token")
endif()
list(APPEND DEFINITIONS_AVX2 ${DEFINITIONS} FN_PROTO_SUFFIX=_avx2)
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS_AVX2}" "avx2")

# x86_64_avx2_log2
set(SRCS
  dlog.c
  alog.c
  log.c)
list(APPEND DEFINITIONS_LOG2 ${DEFINITIONS_AVX2} COMPILING_LOG2)
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS_LOG2}" "avx2_log2")

# x86_64_avx2_log10
set(SRCS
  dlog.c
  alog.c
  log.c)
list(APPEND DEFINITIONS_LOG10 ${DEFINITIONS_AVX2} COMPILING_LOG10)
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS_LOG10}" "avx2_log10")
