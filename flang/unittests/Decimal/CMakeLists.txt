#===-- unittests/Decimal/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_nongtest_unittest(quick-sanity-test
quick-sanity-test.cpp
LINK_LIBS
  flang_rt.unittest
)

# This test is not run by default as it takes a long time to execute.
add_flangrt_nongtest_unittest(thorough-test SLOW_TEST
thorough-test.cpp
LINK_LIBS
  flang_rt.unittest
)
