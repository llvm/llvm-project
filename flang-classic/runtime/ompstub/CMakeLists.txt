#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(OMPSTUB_SRC ompstubs.c)

add_flang_library(ompstub_static ${OMPSTUB_SRC})
if (MSVC)
  set_property(TARGET ompstub_static PROPERTY OUTPUT_NAME libompstub)
else()
  set_property(TARGET ompstub_static PROPERTY OUTPUT_NAME ompstub)
endif()

set(SHARED_LIBRARY TRUE)
add_flang_library(ompstub_shared ${OMPSTUB_SRC})
set_property(TARGET ompstub_shared PROPERTY OUTPUT_NAME ompstub)
if (MSVC)
  set_target_properties(ompstub_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()
set(SHARED_LIBRARY FALSE)

set_target_properties(ompstub_static ompstub_shared
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${FLANG_RTE_LIB_DIR}
  )

