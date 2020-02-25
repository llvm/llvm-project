#===-- lib/Common/CMakeLists.txt -------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_library(FortranCommon
  Fortran.cpp
  Fortran-features.cpp
  default-kinds.cpp
  idioms.cpp
)

install (TARGETS FortranCommon
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
