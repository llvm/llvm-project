#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(astutil 
  astutil.c
  ${UTILS_COMMON_DIR}/utils.cpp
  )

target_include_directories(astutil
  PRIVATE
  ${FLANG_INCLUDE_DIR} 
  ${UTILS_COMMON_DIR}
  ${TOOLS_SHARED_DIR}
  ${FLANG1_SRC_DIR}
  ${FLANG1_INCLUDE_DIR}
  )

# Generate AST headers

add_custom_command( 
  OUTPUT ${UTILS_AST_BIN_DIR}/ast.out.n ${UTILS_AST_BIN_DIR}/ast.h ${UTILS_AST_BIN_DIR}/astdf.h
  COMMAND ${CMAKE_BINARY_DIR}/bin/astutil ${UTILS_AST_DIR}/ast.n ${UTILS_AST_DIR}/ast.in.h ${UTILS_SYMTAB_BIN_DIR}/ast.d ${UTILS_SYMTAB_BIN_DIR}/astdf.d -o -n ${UTILS_AST_BIN_DIR}/ast.out.n ${UTILS_AST_BIN_DIR}/ast.h ${UTILS_AST_BIN_DIR}/astdf.h
  DEPENDS gen_frontend_symini astutil ${UTILS_AST_DIR}/ast.n ${UTILS_AST_DIR}/ast.in.h ${UTILS_SYMTAB_BIN_DIR}/ast.d ${UTILS_SYMTAB_BIN_DIR}/astdf.d
  )

add_custom_target(gen_frontend_ast
  SOURCES ${UTILS_AST_BIN_DIR}/ast.out.n ${UTILS_AST_BIN_DIR}/ast.h ${UTILS_AST_BIN_DIR}/astdf.h
  )
