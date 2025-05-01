#===-- lib/CMakeLists.txt --------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_subdirectory(quadmath)
add_subdirectory(runtime)
if (FLANG_RT_INCLUDE_CUF)
  add_subdirectory(cuda)
endif()

if (FLANG_RT_INCLUDE_TESTS)
  add_subdirectory(Testing)
endif ()
