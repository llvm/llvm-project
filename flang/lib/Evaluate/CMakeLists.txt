include(CheckLibraryExists)

if (LIBPGMATH_DIR)
  # If pgmath library is found, it can be used for constant folding.
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    # pgmath uses _Complex, so only enable linking pgmath with flang in environments
    # that support it (MSVC is OK, pgmath uses _Fcomplex/_Dcomplex there).
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU|MSVC")
      check_cxx_compiler_flag("-Werror -Wc99-extensions" HAS_WC99_EXTENSIONS_FLAG)
      if (HAS_WC99_EXTENSIONS_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c99-extensions")
      endif()
      add_compile_definitions(LINK_WITH_LIBPGMATH)
      message(STATUS "Found libpgmath: ${LIBPGMATH}")
    else()
      message(STATUS "Libpgmath will not be used because C99 complex is not supported.")
    endif()
  else()
    message(STATUS "Libpgmath not found in: ${LIBPGMATH_DIR}")
  endif()
endif()

# Compile-time folding of REAL(16) constants needs a binary128 math library on
# the host. Which one is not a separate question from which one the runtime
# uses: before this was tied to FLANG_RUNTIME_F128_MATH_LIB, the folder was
# enabled purely by libquadmath being installed on the build machine, so every
# flang binary linked it - including one configured for the libm route, and one
# configured with REAL(16) disabled entirely, where the table was unreachable
# dead weight.
#
# Worse, on the libm route the two halves then disagreed about who computes:
# `parameter :: v = sin(x)` folded through sinq while `y = sin(x)` ran through
# sinf128. They agree on every point measured, but nothing makes them agree in
# the last place.
if (FLANG_RUNTIME_F128_MATH_LIB STREQUAL "libquadmath")
  check_library_exists(quadmath sinq "" FOUND_QUADMATH_LIB)
  if (FLANG_INCLUDE_QUADMATH_H AND FOUND_QUADMATH_LIB)
    configure_file("${FLANG_SOURCE_DIR}/cmake/quadmath_wrapper.h.in" "${CMAKE_CURRENT_BINARY_DIR}/quadmath_wrapper.h")
    add_compile_definitions(HAS_QUADMATHLIB)
    set(QUADMATHLIB quadmath)
  else()
    # BEHAVIOUR CHANGE, and a deliberate one. Previously this condition was
    # silent: the folding table was simply not built, configuration carried on,
    # and the resulting compiler evaluated REAL(16) constant expressions at run
    # time instead of folding them - with no message at any point saying so.
    #
    # That silence is only harmless while nobody asked. Here the user has named
    # libquadmath explicitly, so the absence of quadmath.h or of the library is
    # a request that cannot be honoured, and continuing would hand back a
    # compiler quietly missing the capability that was asked for. The same
    # reasoning does not apply when the option is left empty: REAL(16) is then
    # disabled outright and there is nothing to be quiet about.
    #
    # A configuration that relied on the old behaviour - naming libquadmath on
    # a machine that does not have it - will now stop here. The fix is to leave
    # FLANG_RUNTIME_F128_MATH_LIB empty, which is what such a build was
    # effectively getting anyway.
    message(FATAL_ERROR
      "FLANG_RUNTIME_F128_MATH_LIB=libquadmath but quadmath.h or libquadmath "
      "was not found; the compiler could not fold REAL(16) constants")
  endif()
elseif (FLANG_RUNTIME_F128_MATH_LIB STREQUAL "libm")
  # glibc 2.26 and later carry the same functions as *f128 in libm, which is
  # linked anyway. The macro that reveals their prototypes has to be on the
  # command line for the same reason as in flang-rt: no header can guarantee it
  # is seen before <cmath> in every translation unit.
  add_compile_definitions(HAS_LIBMF128_HOST)
  add_compile_definitions(__STDC_WANT_IEC_60559_TYPES_EXT__=1)
endif ()

add_flang_library(FortranEvaluate
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
  fold-designator.cpp
  fold-integer.cpp
  fold-logical.cpp
  fold-real.cpp
  fold-reduction.cpp
  formatting.cpp
  host.cpp
  initial-image.cpp
  integer.cpp
  intrinsics.cpp
  intrinsics-library.cpp
  logical.cpp
  real.cpp
  shape.cpp
  static-data.cpp
  target.cpp
  tools.cpp
  type.cpp
  variable.cpp

  LINK_LIBS
  FortranSupport
  FortranDecimal
  FortranParser
  ${LIBPGMATH}
  ${QUADMATHLIB}

  DISABLE_PCH_REUSE
  PRECOMPILE_HEADERS
  [["flang/Evaluate/common.h"]]
  [["flang/Evaluate/call.h"]]
  [["flang/Evaluate/traverse.h"]]
  [["flang/Evaluate/shape.h"]]
  [["flang/Evaluate/characteristics.h"]]
  [["flang/Evaluate/variable.h"]]
  [["flang/Evaluate/real.h"]]
  [["flang/Evaluate/type.h"]]
  [["flang/Evaluate/integer.h"]]
  [["flang/Evaluate/expression.h"]]
  [["flang/Evaluate/tools.h"]]

  LINK_COMPONENTS
  Support

  DEPENDS
  acc_gen
  omp_gen
)
