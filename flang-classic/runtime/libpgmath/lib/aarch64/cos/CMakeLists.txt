#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler flags and definitions
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_L1")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_L1")

set(SRCS gscos4.c gdcos2.c)

libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")
