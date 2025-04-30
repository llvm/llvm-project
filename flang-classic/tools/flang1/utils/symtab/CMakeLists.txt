#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Generate symbol table headers

add_custom_command(
  OUTPUT ${UTILS_SYMTAB_BIN_DIR}/symtab.out.n
         ${UTILS_SYMTAB_BIN_DIR}/symtab.h
         ${UTILS_SYMTAB_BIN_DIR}/symtabdf.h
         ${UTILS_SYMTAB_BIN_DIR}/symtabdf.c
         ${UTILS_SYMTAB_BIN_DIR}/symnames.h
         ${FLANG1_DOC_BIN_DIR}/symtab.rst
  COMMAND fesymutil ${CMAKE_CURRENT_SOURCE_DIR}/symtab.n
                                            ${CMAKE_CURRENT_SOURCE_DIR}/symtab.in.h
                                      -o -n ${UTILS_SYMTAB_BIN_DIR}/symtab.out.n
                                            ${UTILS_SYMTAB_BIN_DIR}/symtab.h
                                            ${UTILS_SYMTAB_BIN_DIR}/symtabdf.h
                                            ${UTILS_SYMTAB_BIN_DIR}/symtabdf.c
                                            ${UTILS_SYMTAB_BIN_DIR}/symnames.h
                                         -s ${FLANG1_DOC_BIN_DIR}/symtab.rst
  DEPENDS fesymutil ${CMAKE_CURRENT_SOURCE_DIR}/symtab.n
                    ${CMAKE_CURRENT_SOURCE_DIR}/symtab.in.h
  )

add_custom_target(gen_frontend_symtab
  DEPENDS ${UTILS_SYMTAB_BIN_DIR}/symtab.out.n
          ${UTILS_SYMTAB_BIN_DIR}/symtab.h
          ${UTILS_SYMTAB_BIN_DIR}/symtabdf.h
          ${UTILS_SYMTAB_BIN_DIR}/symtabdf.c
          ${UTILS_SYMTAB_BIN_DIR}/symnames.h
          ${FLANG1_DOC_BIN_DIR}/symtab.rst
  )

# Generate symini and ILM headers

add_custom_command(
  OUTPUT ${UTILS_SYMTAB_BIN_DIR}/pd.h
         ${UTILS_SYMTAB_BIN_DIR}/syminidf.h
         ${UTILS_SYMTAB_BIN_DIR}/ast.d
         ${UTILS_SYMTAB_BIN_DIR}/astdf.d
         ${UTILS_SYMTAB_BIN_DIR}/ilmtp.h
         ${FLANG1_DOC_BIN_DIR}/symini.rst
  COMMAND fesymini ${UTILS_SYMTAB_DIR}/symini_ftn.n
                                        -o ${UTILS_SYMTAB_BIN_DIR}/syminidf.h
                                           ${UTILS_SYMTAB_BIN_DIR}/pd.h
                                           ${UTILS_SYMTAB_BIN_DIR}/ast.d
                                           ${UTILS_SYMTAB_BIN_DIR}/astdf.d
                                           ${UTILS_SYMTAB_BIN_DIR}/ilmtp.h
                                        -s ${FLANG1_DOC_BIN_DIR}/symini.rst
  DEPENDS fesymini ${UTILS_SYMTAB_DIR}/symini_ftn.n
  )

add_custom_target(gen_frontend_symini
  SOURCES ${UTILS_SYMTAB_BIN_DIR}/pd.h
          ${UTILS_SYMTAB_BIN_DIR}/syminidf.h
          ${UTILS_SYMTAB_BIN_DIR}/ast.d
          ${UTILS_SYMTAB_BIN_DIR}/astdf.d
          ${UTILS_SYMTAB_BIN_DIR}/ilmtp.h
          ${FLANG1_DOC_BIN_DIR}/symini.rst
  )


add_executable(fesymutil
  ${UTILS_SHARED_DIR}/symutil.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
  )

add_dependencies(fesymutil gen_frontend_error_headers)

target_include_directories(fesymutil
  BEFORE
  PRIVATE
  ${UTILS_SHARED_DIR}
  ${FLANG_INCLUDE_DIR}
  ${UTILS_COMMON_DIR}
  )


target_compile_definitions(fesymutil
  PRIVATE
  PGF90
  PGHPF
  )

add_executable(fesymini
  ${UTILS_SHARED_DIR}/symacc.c
  symini.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
  ${UTILS_SYMTAB_BIN_DIR}/symtabdf.c
  )

add_dependencies(fesymini gen_frontend_symtab)

target_compile_definitions(fesymini
  PRIVATE
  PGF90
  PGHPF
  UTILSYMTAB
  )

target_include_directories(fesymini
  PRIVATE
  ${SHARED_UTILS_INCLUDE_DIR}/symtab
  ${FLANG_INCLUDE_DIR}
  ${UTILS_SHARED_DIR}
  ${UTILS_COMMON_DIR}
  ${UTILS_SYMTAB_BIN_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# Local Variables:
# mode: cmake
# End:
