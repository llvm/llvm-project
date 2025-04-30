#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Build lr and prodstr

add_executable(lr
  lrutils.c
  prstab1.c
  prstab2.c
  prstab3.c
  prstab4.c
  )

add_executable(prodstr
  prodstr.c
  )

# Generate parse tables

add_custom_command(
  OUTPUT ${UTILS_PARSETABLE_BIN_DIR}/gramdf.h ${UTILS_PARSETABLE_BIN_DIR}/gramsm.h ${UTILS_PARSETABLE_BIN_DIR}/gramtk.h ${UTILS_PARSETABLE_BIN_DIR}/proddf.h ${UTILS_PARSETABLE_BIN_DIR}/tokdf.h ${UTILS_PARSETABLE_BIN_DIR}/gramnt.h ${UTILS_PARSETABLE_BIN_DIR}/gram.lis
  # Copy to current build directory, otherwise header will be generated in
  # original directory
  COMMAND ${CMAKE_COMMAND} -E copy ${UTILS_PARSETABLE_DIR}/gram.txt ${UTILS_PARSETABLE_BIN_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${UTILS_PARSETABLE_DIR}/gram.tki ${UTILS_PARSETABLE_BIN_DIR}
  COMMAND ${CMAKE_BINARY_DIR}/bin/lr gram.txt -semant -tokens -list -data -runoff
  COMMAND ${CMAKE_BINARY_DIR}/bin/prodstr gram.txt
  DEPENDS ${UTILS_PARSETABLE_DIR}/gram.tki ${UTILS_PARSETABLE_DIR}/gram.txt
  )

add_custom_target(gen_frontend_parsetable
  SOURCES ${UTILS_PARSETABLE_BIN_DIR}/gramdf.h ${UTILS_PARSETABLE_BIN_DIR}/gramsm.h ${UTILS_PARSETABLE_BIN_DIR}/gramtk.h ${UTILS_PARSETABLE_BIN_DIR}/proddf.h ${UTILS_PARSETABLE_BIN_DIR}/tokdf.h ${UTILS_PARSETABLE_BIN_DIR}/gramnt.h
  )

add_dependencies(gen_frontend_parsetable lr)
add_dependencies(gen_frontend_parsetable prodstr)
add_dependencies(gen_frontend_parsetable gen_frontend_error_headers)
