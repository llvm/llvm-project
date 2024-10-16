#===-- lib/CMakeLists.txt --------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(FortranFloat128Math)
add_subdirectory(flang_rt)
if (FLANG_RT_ENABLE_CUF)
  add_subdirectory(CufRuntime)
endif()

if (FLANG_RT_INCLUDE_TESTS)
  add_subdirectory(Testing)
endif ()
