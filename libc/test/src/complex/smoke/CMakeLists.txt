add_libc_testsuite(libc-complex-smoke-tests PARENT libc-complex-tests)

add_fp_unittest(
  cabs_test
  SUITE
    libc-complex-smoke-tests
  SRCS
    cabs_test.cpp
  HDRS
    ../CAbsTest.h
  DEPENDS
    libc.src.complex.cabs
)

add_fp_unittest(
  cabsf_test
  SUITE
    libc-complex-smoke-tests
  SRCS
    cabsf_test.cpp
  HDRS
    ../CAbsTest.h
  DEPENDS
    libc.src.complex.cabsf
)

add_fp_unittest(
  carg_test
  SUITE
    libc-complex-smoke-tests
  SRCS
    carg_test.cpp
  HDRS
    ../CArgTest.h
  DEPENDS
    libc.src.complex.carg
)

add_fp_unittest(
  cargf_test
  SUITE
    libc-complex-smoke-tests
  SRCS
    cargf_test.cpp
  HDRS
    ../CArgTest.h
  DEPENDS
    libc.src.complex.cargf
)
