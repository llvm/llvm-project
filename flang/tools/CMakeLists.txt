#===-- tools/CMakeLists.txt ------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(f18)
if(FLANG_BUILD_NEW_DRIVER)
  add_subdirectory(flang-driver)
endif()
add_subdirectory(tco)
add_subdirectory(f18-parse-demo)
