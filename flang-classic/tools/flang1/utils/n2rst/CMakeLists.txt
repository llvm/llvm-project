#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(fen2rst
  ${UTILS_SHARED_DIR}/n2rst.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
)

target_include_directories(fen2rst
  PRIVATE
  ${UTILS_COMMON_DIR}
  )

# Convert nroff documentation to Sphinx format.

file(MAKE_DIRECTORY ${FLANG1_DOC_BIN_DIR})

add_custom_command(
  OUTPUT ${FLANG1_DOC_BIN_DIR}/ast.rst
         ${FLANG1_DOC_BIN_DIR}/commopt.rst
         ${FLANG1_DOC_BIN_DIR}/comms.rst
         ${FLANG1_DOC_BIN_DIR}/controller.rst
         ${FLANG1_DOC_BIN_DIR}/dinit.rst
         ${FLANG1_DOC_BIN_DIR}/grammar.rst
         ${FLANG1_DOC_BIN_DIR}/gram.rst
         ${FLANG1_DOC_BIN_DIR}/intro.rst
         ${FLANG1_DOC_BIN_DIR}/outconv.rst
         ${FLANG1_DOC_BIN_DIR}/output.rst
         ${FLANG1_DOC_BIN_DIR}/parser.rst
         ${FLANG1_DOC_BIN_DIR}/scanner.rst
         ${FLANG1_DOC_BIN_DIR}/semant.rst
         ${FLANG1_DOC_BIN_DIR}/transform.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/fen2rst -v ${UTILS_AST_DIR}/ast.n
                                             ${FLANG1_DOC_SRC_DIR}/commopt.n
                                             ${FLANG1_DOC_SRC_DIR}/comms.n
                                             ${FLANG1_DOC_SRC_DIR}/controller.n
                                             ${FLANG1_DOC_SRC_DIR}/dinit.n
                                             ${FLANG1_DOC_SRC_DIR}/grammar.n
                                             ${UTILS_PARSETABLE_BIN_DIR}/gram.lis
                                             ${FLANG1_DOC_SRC_DIR}/intro.n
                                             ${FLANG1_DOC_SRC_DIR}/outconv.n
                                             ${FLANG1_DOC_SRC_DIR}/output.n
                                             ${FLANG1_DOC_SRC_DIR}/parser.n
                                             ${FLANG1_DOC_SRC_DIR}/scanner.n
                                             ${FLANG1_DOC_SRC_DIR}/semant.n
                                             ${FLANG1_DOC_SRC_DIR}/transform.n
  WORKING_DIRECTORY ${FLANG1_DOC_BIN_DIR}
  DEPENDS fen2rst ${UTILS_AST_DIR}/ast.n
                  ${FLANG1_DOC_SRC_DIR}/commopt.n
                  ${FLANG1_DOC_SRC_DIR}/comms.n
                  ${FLANG1_DOC_SRC_DIR}/controller.n
                  ${FLANG1_DOC_SRC_DIR}/dinit.n
                  ${FLANG1_DOC_SRC_DIR}/grammar.n
                  ${UTILS_PARSETABLE_BIN_DIR}/gram.lis
                  ${FLANG1_DOC_SRC_DIR}/intro.n
                  ${FLANG1_DOC_SRC_DIR}/outconv.n
                  ${FLANG1_DOC_SRC_DIR}/output.n
                  ${FLANG1_DOC_SRC_DIR}/parser.n
                  ${FLANG1_DOC_SRC_DIR}/scanner.n
                  ${FLANG1_DOC_SRC_DIR}/semant.n
                  ${FLANG1_DOC_SRC_DIR}/transform.n
  )

add_custom_target(flang1_gen_sphinx_docs
  SOURCES ${FLANG1_DOC_BIN_DIR}/ast.rst
          ${FLANG1_DOC_BIN_DIR}/commopt.rst
          ${FLANG1_DOC_BIN_DIR}/comms.rst
          ${FLANG1_DOC_BIN_DIR}/controller.rst
          ${FLANG1_DOC_BIN_DIR}/dinit.rst
          ${FLANG1_DOC_BIN_DIR}/grammar.rst
          ${FLANG1_DOC_BIN_DIR}/gram.rst
          ${FLANG1_DOC_BIN_DIR}/intro.rst
          ${FLANG1_DOC_BIN_DIR}/outconv.rst
          ${FLANG1_DOC_BIN_DIR}/output.rst
          ${FLANG1_DOC_BIN_DIR}/parser.rst
          ${FLANG1_DOC_BIN_DIR}/scanner.rst
          ${FLANG1_DOC_BIN_DIR}/semant.rst
          ${FLANG1_DOC_BIN_DIR}/transform.rst
  )

add_dependencies(flang1_gen_sphinx_docs gen_frontend_parsetable)

# Local Variables:
# mode: cmake
# End:
