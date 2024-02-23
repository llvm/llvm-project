#===-- runtime/Float128Math/CMakeLists.txt ---------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# FortranFloat128 implements IEEE-754 128-bit float math functions.
# It is a thin wapper and it currently relies on third-party
# libraries available for the target.
# It is distributed as a static library only.
# Fortran programs/libraries that end up linking any of the provided
# will have a dependency on the third-party library that is being
# used for building this FortranFloat128Math library.

if (${FLANG_RUNTIME_F128_MATH_LIB} STREQUAL "libquadmath" OR
    ${FLANG_RUNTIME_F128_MATH_LIB} STREQUAL "quadmath")
  check_include_file(quadmath.h FOUND_QUADMATH_HEADER)
  if(FOUND_QUADMATH_HEADER)
    add_compile_definitions(HAS_QUADMATHLIB)
  else()
    message(FATAL_ERROR
      "FLANG_RUNTIME_F128_MATH_LIB setting requires quadmath.h "
      "to be available: ${FLANG_RUNTIME_F128_MATH_LIB}"
      )
  endif()
else()
  message(FATAL_ERROR
    "Unsupported third-party library for Fortran F128 math runtime: "
    "${FLANG_RUNTIME_F128_MATH_LIB}"
    )
endif()

set(sources
  cabs.cpp
  sin.cpp
  sqrt.cpp
  )

include_directories(AFTER "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_flang_library(FortranFloat128Math STATIC INSTALL_WITH_TOOLCHAIN ${sources})

if (DEFINED MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  add_flang_library(FortranFloat128Math.static STATIC INSTALL_WITH_TOOLCHAIN
    ${sources}
    )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebug)
  add_flang_library(FortranFloat128Math.static_dbg STATIC INSTALL_WITH_TOOLCHAIN
    ${sources}
    )
  add_dependencies(FortranFloat128Math FortranFloat128Math.static
    FortranFloat128Math.static_dbg
    )
endif()
