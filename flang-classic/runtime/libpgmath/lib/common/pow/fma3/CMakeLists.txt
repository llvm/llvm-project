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
      sdpow.cpp
      sspow.cpp
      vdpow2.cpp
      vdpow4.cpp
      vspow4.cpp
      vspow8.cpp)
    libmath_add_object_library("${SRCS}" "${FLAGS} -mfma -mavx2" "${DEFINITIONS}" "")
    set(SRCS
      vdpow8.cpp
      vspow16.cpp)
    set(FLAGS_AVX512 "${FLAGS} -march=skylake-avx512 -mtune=skylake-avx512")
    libmath_add_object_library("${SRCS}" "${FLAGS_AVX512}" "${DEFINITIONS}" "pow-avx512")
    set(FLAGS_AVX512 "${FLAGS} -march=knl -mtune=knl")
    libmath_add_object_library("${SRCS}" "${FLAGS_AVX512}" "${DEFINITIONS}" "pow-knl")
  endif()
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set(SRCS
    sdpow.cpp
    sspow.cpp
    vdpow2.cpp
    vspow4.cpp)
  set_property(SOURCE sdpow.cpp vdpow2.cpp APPEND PROPERTY COMPILE_DEFINITIONS VINT=long)
  set_property(SOURCE sspow.cpp vspow4.cpp APPEND PROPERTY COMPILE_DEFINITIONS VINT=)
  libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set(SRCS
    sdpow.cpp
    sspow.cpp
    vdpow2.cpp)
  set_property(SOURCE sdpow.cpp vdpow2.cpp APPEND PROPERTY COMPILE_DEFINITIONS VINT=long)
  set_property(SOURCE sspow.cpp vspow4.cpp APPEND PROPERTY COMPILE_DEFINITIONS VINT=)
  set_property(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=${LLVM_FLANG_CPU_TARGET} ")
endif()
