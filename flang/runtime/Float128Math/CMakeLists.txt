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

include(CheckLibraryExists)

if (${FLANG_RUNTIME_F128_MATH_LIB} STREQUAL "libquadmath")
  check_include_file(quadmath.h FOUND_QUADMATH_HEADER)
  if(FOUND_QUADMATH_HEADER)
    add_compile_definitions(HAS_QUADMATHLIB)
  else()
    message(FATAL_ERROR
      "FLANG_RUNTIME_F128_MATH_LIB setting requires quadmath.h "
      "to be available: ${FLANG_RUNTIME_F128_MATH_LIB}"
      )
  endif()
elseif (${FLANG_RUNTIME_F128_MATH_LIB} STREQUAL "libm")
  check_library_exists(m sinl "" FOUND_LIBM)
  check_library_exists(m sinf128 "" FOUND_LIBMF128)
  if (FOUND_LIBM)
    add_compile_definitions(HAS_LIBM)
  endif()
  if (FOUND_LIBMF128)
    add_compile_definitions(HAS_LIBMF128)
  endif()
endif()

if (NOT FOUND_QUADMATH_HEADER AND NOT FOUND_LIBM)
  message(FATAL_ERROR
    "Unsupported third-party library for Fortran F128 math runtime: "
    "${FLANG_RUNTIME_F128_MATH_LIB}"
    )
endif()

set(sources
  acos.cpp
  acosh.cpp
  asin.cpp
  asinh.cpp
  atan.cpp
  atan2.cpp
  atanh.cpp
  cabs.cpp
  ceil.cpp
  cos.cpp
  cosh.cpp
  erf.cpp
  erfc.cpp
  exp.cpp
  exponent.cpp
  floor.cpp
  fraction.cpp
  hypot.cpp
  j0.cpp
  j1.cpp
  jn.cpp
  lgamma.cpp
  llround.cpp
  log.cpp
  log10.cpp
  lround.cpp
  mod-real.cpp
  modulo-real.cpp
  nearest.cpp
  norm2.cpp
  pow.cpp
  round.cpp
  rrspacing.cpp
  scale.cpp
  set-exponent.cpp
  sin.cpp
  sinh.cpp
  spacing.cpp
  sqrt.cpp
  tan.cpp
  tanh.cpp
  tgamma.cpp
  trunc.cpp
  y0.cpp
  y1.cpp
  yn.cpp
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
