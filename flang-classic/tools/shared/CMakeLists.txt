#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(SHARED_SOURCES 
  ${CMAKE_CURRENT_SOURCE_DIR}/ccffinfo.c
  ${CMAKE_CURRENT_SOURCE_DIR}/direct.c
  ${CMAKE_CURRENT_SOURCE_DIR}/error.c
  ${CMAKE_CURRENT_SOURCE_DIR}/go.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ilidir.c
  ${CMAKE_CURRENT_SOURCE_DIR}/llmputil.c
  ${CMAKE_CURRENT_SOURCE_DIR}/mall.c
  ${CMAKE_CURRENT_SOURCE_DIR}/miscutil.c
  ${CMAKE_CURRENT_SOURCE_DIR}/pragma.c
  ${CMAKE_CURRENT_SOURCE_DIR}/rtlRtns.c
  ${CMAKE_CURRENT_SOURCE_DIR}/salloc.c
  ${CMAKE_CURRENT_SOURCE_DIR}/mach.c
  ${CMAKE_CURRENT_SOURCE_DIR}/nmeutil.c
  PARENT_SCOPE
)

include_directories($(CMAKE_CURRENT_SOURCE_DIR))
