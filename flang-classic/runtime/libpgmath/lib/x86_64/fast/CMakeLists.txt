#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler flags and definitions
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L2")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L2")

# Set source files
set(SRCS
  fastcdiv.S
  fastmod.S)
libmath_add_object_library("${SRCS}" "${FLAGS} -mtune=skylake-avx512 -march=skylake-avx512" "${DEFINITIONS}" "")

set(TARGET_NAME "mthdecls")
list(APPEND PREPROCESSOR "${CMAKE_C_COMPILER} -E -DMTH_CMPLX_C99_ABI -DMTH_NO_STD_MATH_HDRS -dM ${CMAKE_CURRENT_SOURCE_DIR}/../${TARGET_NAME}.h")
separate_arguments(PREPROCESSOR UNIX_COMMAND "${PREPROCESSOR}")

add_custom_command(OUTPUT ${TARGET_NAME}.i PRE_BUILD
  COMMAND ${PREPROCESSOR} > ${TARGET_NAME}.i
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../${TARGET_NAME}.h)

if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tmp.${TARGET_NAME}.h.bak "#define __MTH_C99_CMPLX_SUFFIX _c99")
    configure_file(${CMAKE_CURRENT_BINARY_DIR}/tmp.${TARGET_NAME}.h.bak ${CMAKE_CURRENT_BINARY_DIR}/tmp.${TARGET_NAME}.h COPYONLY)
else()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tmp.${TARGET_NAME}.h PRE_BUILD
      COMMAND cat ${TARGET_NAME}.i 2>&1 | grep -e "^#define __MTH_C99_CMPLX_SUFFIX" > ${CMAKE_CURRENT_BINARY_DIR}/tmp.${TARGET_NAME}.h
      DEPENDS ${TARGET_NAME}.i)
endif()

add_custom_target(${TARGET_NAME} ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tmp.${TARGET_NAME}.h)

if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
  list(APPEND DEFINITIONS LINUX_ELF)
endif()
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(SRCS
  fastmath.S)
libmath_add_object_library("${SRCS}" "-m64 -O3 -mtune=skylake-avx512 -march=skylake-avx512" "${DEFINITIONS}" "fastmath")
add_dependencies(fastmath mthdecls)
