#===-- unittests/Evaluate/CMakeLists.txt -----------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flang_nongtest_unittest(reshape
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranRuntime
)

add_flang_nongtest_unittest(ISO-Fortran-binding
  FortranEvaluateTesting
  FortranEvaluate
  FortranSemantics
  FortranRuntime
)
