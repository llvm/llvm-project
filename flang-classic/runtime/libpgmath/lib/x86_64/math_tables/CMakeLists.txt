#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

get_property(FLAGS GLOBAL PROPERTY "FLAGS_X8664_L1")
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_X8664_L2")

set(SRCS
  mth_acosdefs.h
  mth_asindefs.h
  mth_atandefs.h
  mth_atan2defs.h
  mth_cosdefs.h
  mth_sindefs.h
  mth_tandefs.h
  mth_coshdefs.h
  mth_sinhdefs.h
  mth_tanhdefs.h
  mth_expdefs.h
  mth_logdefs.h
  mth_log10defs.h
  mth_powdefs.h
  mth_powidefs.h
  mth_sincosdefs.h
  mth_divdefs.h
  mth_sqrtdefs.h
  mth_moddefs.h
  mth_ceildefs.h
  mth_floordefs.h
  )

set(NEW_SRCS)
foreach(file ${SRCS})
  list(APPEND NEW_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
set(SRCS ${NEW_SRCS})
set(DEPENDENCIES "${SRCS}")

string(REPLACE ";" " -D" DEFINITIONS "${DEFINITIONS}")
set(DEFINITIONS "-D${DEFINITIONS}")
string(REPLACE ";" " " SRCS "${SRCS}")
#list(APPEND PREPROCESSOR "${CMAKE_C_COMPILER} -E ${DEFINITIONS} -DPGFLANG ${FLAGS} ${SRCS}")
list(APPEND PREPROCESSOR " -E ${DEFINITIONS} -DPGFLANG ${FLAGS} ${SRCS}")
separate_arguments(PREPROCESSOR UNIX_COMMAND "${PREPROCESSOR}")
list(INSERT PREPROCESSOR 0 "${CMAKE_C_COMPILER}")

# Generate tmp-mth_alldefs.h
set(TARGET_NAME "tmp-mth_alldefs")
add_custom_command(OUTPUT ${TARGET_NAME}.i PRE_BUILD
  COMMAND ${PREPROCESSOR} > ${TARGET_NAME}.i
  DEPENDS "${DEPENDENCIES}")

add_custom_command(OUTPUT ${TARGET_NAME}.check PRE_BUILD
  COMMAND "${SH_PROGRAM}" "${LIBPGMATH_TOOLS_DIR}/tmp-mth-check.sh" ${TARGET_NAME}.i
  COMMAND "${CMAKE_COMMAND}" -E touch ${TARGET_NAME}.check
  DEPENDS "${TARGET_NAME}.i")

add_custom_command(OUTPUT ${TARGET_NAME}.h PRE_BUILD
  COMMAND "${SH_PROGRAM}" "${LIBPGMATH_TOOLS_DIR}/${TARGET_NAME}.sh" ${TARGET_NAME}.i ${TARGET_NAME}.h
  DEPENDS "${TARGET_NAME}.check")

add_custom_target(${TARGET_NAME} ALL
  DEPENDS "${TARGET_NAME}.h")
