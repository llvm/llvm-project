#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler flags and definitions
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_L1")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_L1")

# Set source files, compiler flags and definitions
if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    # Set source files
    set(CORE2_SRCS fxpowi.c pxpowi.c)
    set_property(SOURCE ${CORE2_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 ")
    set(SANDYBRIDGE_SRCS fypowi.c pypowi.c)
    set_property(SOURCE ${SANDYBRIDGE_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-march=sandybridge ")
    set(KNL_SRCS pzpowi.c)
    set_property(SOURCE ${KNL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=knl -march=knl ")
    set(SRCS
      ${CORE2_SRCS}
      ${SANDYBRIDGE_SRCS}
      ${KNL_SRCS})
elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    # Set source files
    set(CORE2_SRCS fxpowi.c pxpowi.c)
    set_property(SOURCE ${CORE2_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2 ")
    set(SANDYBRIDGE_SRCS fypowi.c pypowi.c)
    set_property(SOURCE ${SANDYBRIDGE_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx ")
    set(KNL_SRCS pzpowi.c)
    set_property(SOURCE ${KNL_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx512f -mfma ")
    set(SRCS
      ${CORE2_SRCS}
      ${SANDYBRIDGE_SRCS}
      ${KNL_SRCS})
  endif()
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set(SRCS fxpowi.c pxpowi.c)
  set_property(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=power8 ")
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set(SRCS fxpowi.c gspowi4.c gdpowi2.c pxpowi.c)
  set_property(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=${LLVM_FLANG_CPU_TARGET} ")
else()
  set(SRCS fxpowi.c pxpowi.c)  
endif()

libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")
