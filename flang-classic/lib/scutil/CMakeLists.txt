#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_library(scutil
 default-fperror.c
 host-fp-folding.c
 int128.c
 legacy-folding-api.c
 legacy-ints.c
 lockfile.c
 path-utils.c
 pgnewfil.c
 cpu-stopwatch.c
)

target_include_directories(scutil
  PRIVATE
  ${FLANG_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set_target_properties(scutil
  PROPERTIES
  PREFIX ""
  ARCHIVE_OUTPUT_DIRECTORY ${FLANG_LIB_DIR}
  C_STANDARD 99
)

# Clang does not support FENV_ACCESS
set_source_files_properties(
  host-fp-folding.c
  PROPERTIES
  COMPILE_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas"
)
