add_libc_testsuite(libc_math_unittests)

function(add_math_unittest name)
  cmake_parse_arguments(
    "MATH_UNITTEST"
    "NEED_MPFR" # No optional arguments
    "" # Single value arguments
    "" # Multi-value arguments
    ${ARGN}
  )

  if(MATH_UNITTEST_NEED_MPFR)
    if(NOT LIBC_TESTS_CAN_USE_MPFR)
      message("WARNING: Math test ${name} will be skipped as MPFR library is not available.")
      return()
    endif()
  endif()

  add_libc_unittest(${name} ${MATH_UNITTEST_UNPARSED_ARGUMENTS})
  if(MATH_UNITTEST_NEED_MPFR)
    get_fq_target_name(${name} fq_target_name)
    target_link_libraries(${fq_target_name} PRIVATE libcMPFRWrapper -lmpfr -lgmp)
  endif()
endfunction(add_math_unittest)

add_math_unittest(
  cosf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    cosf_test.cpp
  HDRS
    sdcomp26094.h
  DEPENDS
    libc.include.errno
    libc.src.math.cosf
    libc.utils.CPP.standalone_cpp
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  sinf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    sinf_test.cpp
  HDRS
    sdcomp26094.h
  DEPENDS
    libc.include.errno
    libc.src.math.sinf
    libc.utils.CPP.standalone_cpp
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  sincosf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    sincosf_test.cpp
  HDRS
    sdcomp26094.h
  DEPENDS
    libc.include.errno
    libc.src.math.sincosf
    libc.utils.CPP.standalone_cpp
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  fabs_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    fabs_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.fabs
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  fabsf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    fabsf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.fabsf
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  trunc_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    trunc_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.trunc
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  truncf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    truncf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.truncf
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  ceil_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    ceil_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.ceil
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  ceilf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    ceilf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.ceilf
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  floor_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    floor_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.floor
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  floorf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    floorf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.floorf
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  round_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    round_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.round
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  roundf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    roundf_test.cpp
  DEPENDS
    libc.include.math
    libc.src.math.roundf
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  expf_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    expf_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.math
    libc.src.math.expf
    libc.utils.FPUtil.fputil
)

add_math_unittest(
  exp2f_test
  NEED_MPFR
  SUITE
    libc_math_unittests
  SRCS
    exp2f_test.cpp
  DEPENDS
    libc.include.errno
    libc.include.math
    libc.src.math.exp2f
    libc.utils.FPUtil.fputil
)
