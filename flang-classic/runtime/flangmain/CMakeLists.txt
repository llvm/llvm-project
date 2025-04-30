#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

enable_language(C)

# Entry point for Fortran programs

add_flang_library(flangmain
  STATIC
  flangmain.c
  )

set_property(
 SOURCE flangmain.c
 PROPERTY COMPILE_DEFINITIONS 
  TM_I8
  PGF90
  KANJI
 )

set_property(
  SOURCE flangmain.c PROPERTY
  COMPILE_FLAGS -Wno-implicit-function-declaration
  )

set_target_properties(flangmain
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${FLANG_RTE_LIB_DIR}
  )
  
# install step should move from FLANG_LIB_DIR to ???
