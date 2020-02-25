#===-- test/Decimal/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_executable(quick-sanity-test
  quick-sanity-test.cpp
)

target_link_libraries(quick-sanity-test
  FortranDecimal
)

add_executable(thorough-test
  thorough-test.cpp
)

target_link_libraries(thorough-test
  FortranDecimal
)

add_test(Sanity quick-sanity-test)
