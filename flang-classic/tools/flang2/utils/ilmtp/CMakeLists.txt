#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_definitions("-DILMTOOLBUILD")

add_executable(ilmtp
  ilmtp.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
  )

target_include_directories(ilmtp
  PRIVATE
  ${UTILS_COMMON_DIR}
  ${FLANG_TOOLS_DIR}
  )

# Generate ILM tables

add_custom_command(
  OUTPUT ${UTILS_ILM_BIN_DIR}/ilmtpdf.h
         ${UTILS_ILM_BIN_DIR}/ilmtp.h
         ${UTILS_ILM_BIN_DIR}/ilmtp.n1
         ${FLANG2_DOC_BIN_DIR}/ilmtp.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/ilmtp -I${UTILS_ILI_DIR} -l scftn
                                                 ${ARCH_DEP_ILM_DIR}/ilmtp.n
                                                 ${ARCH_DEP_ILI_DIR}/ilitp.n
                                              -o ${UTILS_ILM_BIN_DIR}/ilmtpdf.h
                                                 ${UTILS_ILM_BIN_DIR}/ilmtp.h
                                              -n ${UTILS_ILM_BIN_DIR}/ilmtp.n1
                                              -s ${FLANG2_DOC_BIN_DIR}/ilmtp.rst
  DEPENDS ilmtp ${ARCH_DEP_ILM_DIR}/ilmtp.n
                ${ARCH_DEP_ILM_DIR}/ilmtp_atomic.n
                ${ARCH_DEP_ILM_DIR}/ilmtp_longdouble.n
                ${ARCH_DEP_ILI_DIR}/ilitp.n
  )

add_custom_target(gen_backend_ilm
  SOURCES ${UTILS_ILM_BIN_DIR}/ilmtpdf.h
          ${UTILS_ILM_BIN_DIR}/ilmtp.h
          ${UTILS_ILM_BIN_DIR}/ilmtp.n1
          ${FLANG2_DOC_BIN_DIR}/ilmtp.rst
  )

# Local Variables:
# mode: cmake
# End:
