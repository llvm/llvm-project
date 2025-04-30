#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

add_executable(machar
  ${UTILS_SHARED_DIR}/machar.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
  )

target_include_directories(machar
  PRIVATE
  ${FLANG_INCLUDE_DIR}
  ${UTILS_COMMON_DIR}
  )

target_compile_definitions(machar
  PRIVATE
  FE90
  )

# Generate Machar headers

add_custom_command(
  OUTPUT ${UTILS_MACHAR_BIN_DIR}/machar.h
         ${UTILS_MACHAR_BIN_DIR}/machardf.h
         ${FLANG1_DOC_BIN_DIR}/machar.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/machar ${UTILS_MACHAR_DIR}/machar.n
                                         ${UTILS_SYMTAB_BIN_DIR}/symtab.h
                                      -o ${UTILS_MACHAR_BIN_DIR}/machar.h
                                         ${UTILS_MACHAR_BIN_DIR}/machardf.h
                                      -s ${FLANG1_DOC_BIN_DIR}/machar.rst
  DEPENDS gen_frontend_symtab machar ${UTILS_MACHAR_DIR}/machar.n ${UTILS_SYMTAB_BIN_DIR}/symtab.h
  )

add_custom_target(gen_frontend_machar
  SOURCES ${UTILS_MACHAR_BIN_DIR}/machar.h
          ${UTILS_MACHAR_BIN_DIR}/machardf.h
          ${FLANG1_DOC_BIN_DIR}/machar.rst
  )

# Local Variables:
# mode: cmake
# End:
