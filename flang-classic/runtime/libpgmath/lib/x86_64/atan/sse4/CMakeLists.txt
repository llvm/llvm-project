#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Set compiler definitions and flags
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L2")
get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L2")

# Set source files
set(SRCS
  fvdatan2.S
  fvsatan4.S)
libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")
