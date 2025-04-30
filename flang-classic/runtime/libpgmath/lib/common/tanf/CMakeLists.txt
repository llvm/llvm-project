#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler flags and definitions
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_L1")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_L1")

# Set source files and compilation flags
set(COREAVX2_SRCS fs_tan_1_avx2.cpp fs_tan_4_avx2.cpp fs_tan_8_avx2.cpp)
set_property(SOURCE ${COREAVX2_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=core-avx2 -march=core-avx2 ")
set(SKYLAKE_SRCS fs_tan_16_avx512.cpp)
set_property(SOURCE ${SKYLAKE_SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=skylake-avx512 -march=skylake-avx512 ")

# Set source files
set(SRCS
  ${COREAVX2_SRCS}
  ${SKYLAKE_SRCS})
set_property(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_FLAGS "-Wno-attributes ")
set_property(SOURCE ${SRCS} APPEND_STRING PROPERTY COMPILE_DEFINITIONS "PGI ")

libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")
