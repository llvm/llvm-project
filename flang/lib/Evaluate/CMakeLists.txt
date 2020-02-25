#===-- lib/Evaluate/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_library(FortranEvaluate
  call.cpp
  characteristics.cpp
  check-expression.cpp
  common.cpp
  complex.cpp
  constant.cpp
  expression.cpp
  fold.cpp
  fold-character.cpp
  fold-complex.cpp
  fold-integer.cpp
  fold-logical.cpp
  fold-real.cpp
  formatting.cpp
  host.cpp
  integer.cpp
  intrinsics.cpp
  intrinsics-library.cpp
  logical.cpp
  real.cpp
  shape.cpp
  static-data.cpp
  tools.cpp
  type.cpp
  variable.cpp
)

target_link_libraries(FortranEvaluate
  FortranCommon
  FortranDecimal
  FortranParser
  m
)

install (TARGETS FortranEvaluate
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if (LIBPGMATH_DIR)
  # If pgmath library is found, it can be used for constant folding.
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    add_compile_definitions(LINK_WITH_LIBPGMATH)
    target_link_libraries(FortranEvaluate ${LIBPGMATH})
    message(STATUS "Found libpgmath: ${LIBPGMATH}")
  else()
    message(STATUS "Libpgmath not found in: ${LIBPGMATH_DIR}")
  endif()
endif()
