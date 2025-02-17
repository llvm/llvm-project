#===-- unittests/Runtime/CUDA/CMakeLists.txt -------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_unittest(FlangCufRuntimeTests
  Allocatable.cpp
  AllocatorCUF.cpp
  Memory.cpp
)

target_link_libraries(FlangCufRuntimeTests
  PRIVATE
  flang_rt.cuda.default
)
