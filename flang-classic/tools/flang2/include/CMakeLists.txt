#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Generate platform header

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/platform.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/platform.h
  @ONLY)

# Generate error message headers

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/errmsgdf.h
         ${CMAKE_CURRENT_BINARY_DIR}/new_errmsg.n
         ${FLANG2_DOC_BIN_DIR}/errmsg.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/errmsggen ${FLANG_INCLUDE_DIR}/Error/errmsg-accel.n
                                            ${FLANG_INCLUDE_DIR}/Error/errmsg-common.n
                                            ${FLANG_INCLUDE_DIR}/Error/errmsg-hammer.n
                                            ${FLANG_INCLUDE_DIR}/Error/errmsg.n
                                         -o ${CMAKE_CURRENT_BINARY_DIR}/errmsgdf.h
                                         -e ${CMAKE_CURRENT_BINARY_DIR}/new_errmsg.n
                                         -s ${FLANG2_DOC_BIN_DIR}/errmsg.rst
  DEPENDS errmsggen ${FLANG_INCLUDE_DIR}/Error/errmsg-accel.n
                    ${FLANG_INCLUDE_DIR}/Error/errmsg-common.n
                    ${FLANG_INCLUDE_DIR}/Error/errmsg-hammer.n
                    ${FLANG_INCLUDE_DIR}/Error/errmsg.n
  )

add_custom_target(gen_backend_error_headers
  SOURCES ${CMAKE_CURRENT_BINARY_DIR}/errmsgdf.h
          ${CMAKE_CURRENT_BINARY_DIR}/new_errmsg.n
          ${FLANG2_DOC_BIN_DIR}/errmsg.rst
  )

# Local Variables:
# mode: cmake
# End:
