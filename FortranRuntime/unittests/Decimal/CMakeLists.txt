#===-- unittests/Decimal/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_flang_nongtest_unittest(quick-sanity-test
  FortranDecimal
)

# This test is not run by default as it takes a long time to execute.
add_flang_nongtest_unittest(thorough-test
  SLOW_TEST
  FortranDecimal
)
