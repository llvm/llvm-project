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
         ${UTILS_SYMTAB_BIN_DIR}/symtabdf.cpp
         ${UTILS_SYMTAB_BIN_DIR}/symnames.h
         ${FLANG2_DOC_BIN_DIR}/symtab.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/fsymutil ${CMAKE_CURRENT_SOURCE_DIR}/symtab.n
                                           ${CMAKE_CURRENT_SOURCE_DIR}/symtab.in.h
                                     -o -n ${UTILS_SYMTAB_BIN_DIR}/symtab.out.n
                                           ${UTILS_SYMTAB_BIN_DIR}/symtab.h
                                           ${UTILS_SYMTAB_BIN_DIR}/symtabdf.h
                                           ${UTILS_SYMTAB_BIN_DIR}/symtabdf.cpp
                                           ${UTILS_SYMTAB_BIN_DIR}/symnames.h
                                        -s ${FLANG2_DOC_BIN_DIR}/symtab.rst
  DEPENDS fsymutil ${CMAKE_CURRENT_SOURCE_DIR}/symtab.n ${CMAKE_CURRENT_SOURCE_DIR}/symtab.in.h
  )

add_custom_target(gen_backend_symtab
  DEPENDS ${UTILS_SYMTAB_BIN_DIR}/symtab.out.n
          ${UTILS_SYMTAB_BIN_DIR}/symtab.h
          ${UTILS_SYMTAB_BIN_DIR}/symtabdf.h
          ${UTILS_SYMTAB_BIN_DIR}/symtabdf.cpp
          ${UTILS_SYMTAB_BIN_DIR}/symnames.h
          ${FLANG2_DOC_BIN_DIR}/symtab.rst
  )

add_custom_command(
  OUTPUT ${UTILS_SYMTAB_BIN_DIR}/pd.h
         ${UTILS_SYMTAB_BIN_DIR}/syminidf.h
         ${FLANG2_DOC_BIN_DIR}/symini.rst
  COMMAND ${CMAKE_BINARY_DIR}/bin/fsymini ${UTILS_SYMTAB_DIR}/symini_ftn.n
                                          ${ARCH_DEP_ILM_DIR}/ilmtp.n
                                       -o ${UTILS_SYMTAB_BIN_DIR}/syminidf.h
                                          ${UTILS_SYMTAB_BIN_DIR}/pd.h
                                       -s ${FLANG2_DOC_BIN_DIR}/symini.rst
  DEPENDS fsymini ${UTILS_SYMTAB_DIR}/symini_ftn.n ${ARCH_DEP_ILM_DIR}/ilmtp.n
  )

add_custom_target(gen_backend_symini
  SOURCES ${UTILS_SYMTAB_BIN_DIR}/pd.h
          ${UTILS_SYMTAB_BIN_DIR}/syminidf.h
          ${FLANG2_DOC_BIN_DIR}/symini.rst
  )

add_executable(fsymutil
  ${UTILS_SHARED_DIR}/symutil.cpp
  ${UTILS_COMMON_DIR}/utils.cpp
  )

add_dependencies(fsymutil gen_backend_error_headers)

target_include_directories(fsymutil
  BEFORE
  PRIVATE
  ${UTILS_SHARED_DIR}
  ${FLANG_INCLUDE_DIR}
  ${UTILS_COMMON_DIR}
  )

add_executable(fsymini
  ${UTILS_SHARED_DIR}/symacc.c
  ${UTILS_SHARED_DIR}/symacc.h
  symini.cpp
  ${UTILS_COMMON_DIR}/utils.h
  ${UTILS_COMMON_DIR}/utils.cpp
  ${UTILS_SYMTAB_BIN_DIR}/symtab.h
  ${UTILS_SYMTAB_BIN_DIR}/symtabdf.h
  ${UTILS_SYMTAB_BIN_DIR}/symtabdf.cpp
  ${UTILS_SYMTAB_BIN_DIR}/symnames.h
  )

add_dependencies(fsymini
  gen_backend_symtab
  )

target_compile_definitions(fsymini
  PRIVATE
  UTILSYMTAB
  )

target_include_directories(fsymini
  BEFORE
  PRIVATE
  ${SHARED_UTILS_INCLUDE_DIR}/symtab
  ${FLANG_INCLUDE_DIR}
  ${UTILS_SHARED_DIR}
  ${UTILS_COMMON_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${UTILS_SYMTAB_BIN_DIR}
  )

# Local Variables:
# mode: cmake
# End:
