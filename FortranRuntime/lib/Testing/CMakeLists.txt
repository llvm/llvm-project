#===-- lib/Testing/CMakeLists.txt ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

set(public_headers "")
file(GLOB_RECURSE public_headers
  "${FLANGRUNTIME_SOURCE_DIR}/lib/Testing/*.h"
)

add_fortranruntime_library(NonGTestTesting EXCLUDE_FROM_ALL LINK_TO_LLVM
  testing.cpp
  fp-testing.cpp
 ADDITIONAL_HEADERS
   ${public_headers}
)

