#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


set(FLANG1_SRC_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/flang1exe)
set(FLANG1_DOC_SRC_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/docs)
set(UTILS_SYMTAB_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/utils/symtab)
set(UTILS_AST_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/utils/ast)
set(UTILS_MACHAR_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/utils/machar)
set(UTILS_PARSETABLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils/prstab)

set(UTILS_AST_BIN_DIR        ${CMAKE_CURRENT_BINARY_DIR}/utils/ast)
set(UTILS_SYMTAB_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/utils/symtab)
set(UTILS_MACHAR_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/utils/machar)
set(UTILS_PARSETABLE_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/utils/prstab)
set(FLANG1_DOC_BIN_DIR       ${CMAKE_CURRENT_BINARY_DIR}/../../docs/sphinx/flang1)
set(FLANG1_INCLUDE_DIR       ${CMAKE_CURRENT_BINARY_DIR}/include)

include_directories(${FLANG1_INCLUDE_DIR})

option(FLANG_OPENMP_GPU_NVIDIA "Enable OpenMP Accelerator Offload." OFF)
if (FLANG_OPENMP_GPU_NVIDIA)
    add_definitions("-DOMP_OFFLOAD_LLVM")
endif()

add_subdirectory(include)
add_subdirectory(utils)
add_subdirectory(flang1exe)

# Local Variables:
# mode: cmake
# End:
