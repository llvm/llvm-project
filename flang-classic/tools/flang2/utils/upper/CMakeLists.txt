#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(upperl
  upperl.c
  )

# Generate upper tables.
# FIXME: Preprocessing and sorting should be part of add_custom_command below.
get_property(cdefs DIRECTORY PROPERTY COMPILE_DEFINITIONS)
list(TRANSFORM cdefs PREPEND "-D")

if (NOT MSVC OR "${CMAKE_C_COMPILER_FRONTEND_VARIANT}" STREQUAL "GNU")
  set(COMPILER_ARGUMENTS -E -P -x c)
else()
  set(COMPILER_ARGUMENTS /E /TC)
endif()

execute_process(COMMAND "${CMAKE_C_COMPILER}" ${COMPILER_ARGUMENTS} ${cdefs} ${UTILS_UPPER_DIR}/upperilm.in
  WORKING_DIRECTORY ${UTILS_UPPER_BIN_DIR}
  RESULT_VARIABLE cpp_result
  OUTPUT_VARIABLE cpp_output
  ERROR_QUIET)
if(cpp_result EQUAL 0)
  string(REPLACE "\n" ";" UPPERILM_H_CONTENTS ${cpp_output})
  list(SORT UPPERILM_H_CONTENTS)
  # Skip all lines which starts with '#'
  list(FILTER UPPERILM_H_CONTENTS EXCLUDE REGEX "^#")
  list(JOIN UPPERILM_H_CONTENTS "\n" UPPERILM_H_CONTENTS_SORTED)
  file(WRITE ${UTILS_UPPER_BIN_DIR}/upperilm.sort "${UPPERILM_H_CONTENTS_SORTED}\n")
else()
  message(FATAL_ERROR "Preprocessing of upperilm.in failed; aborting.")
endif()

add_custom_command( 
  OUTPUT ${UTILS_UPPER_BIN_DIR}/upperilm.h
  COMMAND ${CMAKE_BINARY_DIR}/bin/upperl ${UTILS_UPPER_BIN_DIR}/upperilm.sort ${UTILS_UPPER_BIN_DIR}/upperilm.h
  DEPENDS upperl ${UTILS_UPPER_DIR}/upperilm.in
  )

add_custom_target(gen_backend_upper
  SOURCES ${UTILS_UPPER_BIN_DIR}/upperilm.h
  )
