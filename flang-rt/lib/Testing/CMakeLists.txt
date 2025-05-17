#===-- lib/Testing/CMakeLists.txt ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

set(public_headers "")
file(GLOB_RECURSE public_headers
  "${FLANG_SOURCE_DIR}/lib/Testing/*.h"
)

add_flangrt_library(NonGTestTesting EXCLUDE_FROM_ALL LINK_TO_LLVM
      ${FLANG_SOURCE_DIR}/lib/Testing/testing.cpp
      ${FLANG_SOURCE_DIR}/lib/Testing/fp-testing.cpp

    ADDITIONAL_HEADERS
      ${public_headers}
  )
