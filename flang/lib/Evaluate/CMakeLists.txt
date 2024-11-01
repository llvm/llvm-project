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

check_include_file(quadmath.h FOUND_QUADMATH_HEADER)
check_library_exists(quadmath sinq "" FOUND_QUADMATH_LIB)
if(FOUND_QUADMATH_HEADER AND FOUND_QUADMATH_LIB)
  add_compile_definitions(HAS_QUADMATHLIB)
  set(QUADMATHLIB quadmath)
endif()

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
  FortranCommon
  FortranDecimal
  FortranParser
  ${LIBPGMATH}
  ${QUADMATHLIB}

  LINK_COMPONENTS
  Support

  DEPENDS
  acc_gen
  omp_gen
)
