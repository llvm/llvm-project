#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(FLANG_INCLUDE_DIR ${FLANG_SOURCE_DIR}/include/flang)
set(SHARED_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(TOOLS_SHARED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shared)
set(UTILS_SHARED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shared/utils)
set(UTILS_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shared/utils/common)

if( ${TARGET_ARCHITECTURE} STREQUAL "aarch64" )
  add_definitions(-DLLVM_ENABLE_FFI=false)
endif()

add_subdirectory(shared)
add_subdirectory(flang1)
add_subdirectory(flang2)
