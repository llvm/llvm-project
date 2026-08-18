#===-- lib/openmp/CMakeLists.txt --------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_library(flang_rt.openmp STATIC SHARED
  omp_alloc.cpp
  omp_util.cpp
  INSTALL_WITH_TOOLCHAIN
)

