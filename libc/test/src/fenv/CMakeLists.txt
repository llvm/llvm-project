add_libc_testsuite(libc_fenv_unittests)

add_libc_unittest(
  rounding_mode_test
  SUITE
    libc_fenv_unittests
  SRCS
    rounding_mode_test.cpp
  DEPENDS
    libc.src.fenv.fegetround
    libc.src.fenv.fesetround
)

add_libc_unittest(
  exception_status_test
  SUITE
    libc_fenv_unittests
  SRCS
    exception_status_test.cpp
  DEPENDS
    libc.src.fenv.feclearexcept
    libc.src.fenv.feraiseexcept
    libc.src.fenv.fetestexcept
)
