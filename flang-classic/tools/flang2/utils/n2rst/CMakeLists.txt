#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(n2rst
  ${UTILS_SHARED_DIR}/n2rst.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
)

target_include_directories(n2rst
  PRIVATE
  ${UTILS_COMMON_DIR}
  )

# Convert nroff documentation to Sphinx format.

file(MAKE_DIRECTORY ${FLANG2_DOC_BIN_DIR})

add_custom_command(
  OUTPUT ${FLANG2_DOC_BIN_DIR}/coding.rst
         ${FLANG2_DOC_BIN_DIR}/controller.rst
         ${FLANG2_DOC_BIN_DIR}/dinit.rst
         ${FLANG2_DOC_BIN_DIR}/error.rst
         ${FLANG2_DOC_BIN_DIR}/expander.rst
         ${FLANG2_DOC_BIN_DIR}/fin.rst
         ${FLANG2_DOC_BIN_DIR}/ili.rst
         ${FLANG2_DOC_BIN_DIR}/ilitp.rst
         ${FLANG2_DOC_BIN_DIR}/ilm.rst
         ${FLANG2_DOC_BIN_DIR}/intro.rst
         ${FLANG2_DOC_BIN_DIR}/register.rst
         ${FLANG2_DOC_BIN_DIR}/xflag.rst
         ${FLANG2_DOC_BIN_DIR}/xref.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/n2rst -v ${FLANG2_DOC_SRC_DIR}/coding.n
                                           ${FLANG2_DOC_SRC_DIR}/controller.n
                                           ${FLANG2_DOC_SRC_DIR}/dinit.n
                                           ${FLANG2_DOC_SRC_DIR}/error.n
                                           ${FLANG2_DOC_SRC_DIR}/expander.n
                                           ${FLANG2_DOC_SRC_DIR}/fin.n
                                           ${FLANG2_DOC_SRC_DIR}/ili.n
                                           ${ARCH_DEP_ILI_DIR}/ilitp.n
                                           ${FLANG2_DOC_SRC_DIR}/ilm.n
                                           ${FLANG2_DOC_SRC_DIR}/intro.n
                                           ${FLANG2_DOC_SRC_DIR}/register.n
                                           ${FLANG2_DOC_SRC_DIR}/xflag.n
                                           ${FLANG2_DOC_SRC_DIR}/xref.n
  WORKING_DIRECTORY ${FLANG2_DOC_BIN_DIR}
  DEPENDS n2rst ${FLANG2_DOC_SRC_DIR}/coding.n
                ${FLANG2_DOC_SRC_DIR}/controller.n
                ${FLANG2_DOC_SRC_DIR}/dinit.n
                ${FLANG2_DOC_SRC_DIR}/error.n
                ${FLANG2_DOC_SRC_DIR}/expander.n
                ${FLANG2_DOC_SRC_DIR}/fin.n
                ${FLANG2_DOC_SRC_DIR}/ili.n
                ${ARCH_DEP_ILI_DIR}/ilitp.n
                ${FLANG2_DOC_SRC_DIR}/ilm.n
                ${FLANG2_DOC_SRC_DIR}/intro.n
                ${FLANG2_DOC_SRC_DIR}/register.n
                ${FLANG2_DOC_SRC_DIR}/xflag.n
                ${FLANG2_DOC_SRC_DIR}/xref.n
  )

add_custom_target(flang2_gen_sphinx_docs
  SOURCES ${FLANG2_DOC_BIN_DIR}/coding.rst
          ${FLANG2_DOC_BIN_DIR}/controller.rst
          ${FLANG2_DOC_BIN_DIR}/dinit.rst
          ${FLANG2_DOC_BIN_DIR}/error.rst
          ${FLANG2_DOC_BIN_DIR}/expander.rst
          ${FLANG2_DOC_BIN_DIR}/fin.rst
          ${FLANG2_DOC_BIN_DIR}/ili.rst
          ${FLANG2_DOC_BIN_DIR}/ilitp.rst
          ${FLANG2_DOC_BIN_DIR}/ilm.rst
          ${FLANG2_DOC_BIN_DIR}/intro.rst
          ${FLANG2_DOC_BIN_DIR}/register.rst
          ${FLANG2_DOC_BIN_DIR}/xflag.rst
          ${FLANG2_DOC_BIN_DIR}/xref.rst
  )

# Local Variables:
# mode: cmake
# End:
