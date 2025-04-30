#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(ilitp
  ilitp.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
  )

target_compile_definitions(ilitp
  PRIVATE
  ILITP_UTIL
  PGF90=1
  )

target_include_directories(ilitp
  PRIVATE
  ${TOOLS_SHARED_DIR}
  ${FLANG2_SRC_DIR}
  ${FLANG_INCLUDE_DIR}
  ${UTILS_COMMON_DIR}
  ) 

# Generate ILI tables

add_custom_command( 
  OUTPUT ${UTILS_ILI_BIN_DIR}/schinfo.h ${UTILS_ILI_BIN_DIR}/ilinfodf.h ${UTILS_ILI_BIN_DIR}/iliatt.h
  COMMAND ${CMAKE_BINARY_DIR}/bin/ilitp -I${UTILS_ILI_DIR} ${ARCH_DEP_ILI_DIR}/ilitp.n
  DEPENDS ilitp
          ${UTILS_ILI_DIR}/ilitp_atomic.n
          ${ARCH_DEP_ILI_DIR}/ilitp.n
          ${ARCH_DEP_ILI_DIR}/ilitp_longdouble.n
  )

add_custom_target(gen_backend_ili
  SOURCES ${UTILS_ILI_BIN_DIR}/schinfo.h ${UTILS_ILI_BIN_DIR}/ilinfodf.h ${UTILS_ILI_BIN_DIR}/iliatt.h
  )
