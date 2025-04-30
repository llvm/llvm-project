#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler definitions and flags
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_L2")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_L2")

set(SRCS)

# Set source files, compiler flags and definitions
if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux|Darwin|Windows")
    set(SRCS
      fdlog1.cpp
      fdlog2.cpp
      fdlog4.cpp
      fslog1.cpp
      fslog4.cpp
      fslog8.cpp)
    # Additional files with custom compilation flags
    set(SRCS_ADD
      rslog1.cpp
      rslog4.cpp
      rslog8.cpp)
    set_property(SOURCE ${SRCS_ADD} APPEND PROPERTY COMPILE_DEFINITIONS PRECISION=2)
    set(SRCS
      ${SRCS}
      ${SRCS_ADD})
    libmath_add_object_library("${SRCS}" "${FLAGS} -mfma -mavx2" "${DEFINITIONS}" "")
    set(SRCS
      fdlog8.cpp
      fslog16.cpp
      rslog16.cpp)
    set(FLAGS_AVX512 "${FLAGS} -march=skylake-avx512 -mtune=skylake-avx512")
    libmath_add_object_library("${SRCS}" "${FLAGS_AVX512}" "${DEFINITIONS}" "log-avx512")
    set(FLAGS_AVX512 "${FLAGS} -march=knl -mtune=knl")
    libmath_add_object_library("${SRCS}" "${FLAGS_AVX512}" "${DEFINITIONS}" "log-knl")
  endif()
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set(INT_SRCS
    fslog1.cpp
    fslog4.cpp
    rslog1.cpp
    rslog4.cpp)
  set(LONG_SRCS
    fdlog1.cpp
    fdlog2.cpp)
  set(SRCS
    ${INT_SRCS}
    ${LONG_SRCS})
  set_property(SOURCE ${INT_SRCS} APPEND PROPERTY COMPILE_DEFINITIONS VINT=)
  set_property(SOURCE ${LONG_SRCS} APPEND PROPERTY COMPILE_DEFINITIONS VINT=long)
  libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set(INT_SRCS
    fslog1.cpp
    fslog4.cpp
    rslog1.cpp
    rslog4.cpp)
  set(LONG_SRCS
    fdlog1.cpp
    fdlog2.cpp)
  set(SRCS
    ${INT_SRCS}
    ${LONG_SRCS})
  set_property(SOURCE ${INT_SRCS} APPEND PROPERTY COMPILE_DEFINITIONS VINT=)
  set_property(SOURCE ${LONG_SRCS} APPEND PROPERTY COMPILE_DEFINITIONS VINT=long)
  set_property(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=${LLVM_FLANG_CPU_TARGET} ")
endif()

