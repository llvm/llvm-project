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
    libc.utils.FPUtil.fputil
)

add_libc_unittest(
  getenv_and_setenv_test
  SUITE
    libc_fenv_unittests
  SRCS
    getenv_and_setenv_test.cpp
  DEPENDS
    libc.src.fenv.fegetenv
    libc.src.fenv.fesetenv
    libc.utils.FPUtil.fputil
)

add_libc_unittest(
  exception_flags_test
  SUITE
    libc_fenv_unittests
  SRCS
    exception_flags_test.cpp
  DEPENDS
    libc.src.fenv.fegetexceptflag
    libc.src.fenv.fesetexceptflag
    libc.utils.FPUtil.fputil
)

add_libc_unittest(
  feupdateenv_test
  SUITE
    libc_fenv_unittests
  SRCS
    feupdateenv_test.cpp
  DEPENDS
    libc.include.signal
    libc.src.fenv.feupdateenv
    libc.utils.FPUtil.fputil
)

add_libc_unittest(
  feclearexcept_test
  SUITE
    libc_fenv_unittests
  SRCS
    feclearexcept_test.cpp
  DEPENDS
    libc.src.fenv.feclearexcept
    libc.utils.FPUtil.fputil
)

if (NOT LLVM_USE_SANITIZER)
  # Sanitizers don't like SIGFPE. So, we will run the 
  # tests which raise SIGFPE only in non-sanitizer builds.
  add_libc_unittest(
    enabled_exceptions_test
    SUITE
      libc_fenv_unittests
    SRCS
      enabled_exceptions_test.cpp
    DEPENDS
      libc.include.signal
      libc.src.fenv.feclearexcept
      libc.src.fenv.feraiseexcept
      libc.src.fenv.fetestexcept
      libc.utils.FPUtil.fputil
  )

  add_libc_unittest(
    feholdexcept_test
    SUITE
      libc_fenv_unittests
    SRCS
      feholdexcept_test.cpp
    DEPENDS
      libc.include.signal
      libc.src.fenv.feholdexcept
      libc.utils.FPUtil.fputil
  )
endif()
