add_custom_target(libc_fenv_tests)

add_libc_test(
  rounding_mode_test
  SUITE
    libc_fenv_tests
  SRCS
    rounding_mode_test.cpp
  DEPENDS
    libc.src.fenv.fegetround
    libc.src.fenv.fesetround
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  exception_status_test
  SUITE
    libc_fenv_tests
  SRCS
    exception_status_test.cpp
  DEPENDS
    libc.src.fenv.feclearexcept
    libc.src.fenv.feraiseexcept
    libc.src.fenv.fesetexcept
    libc.src.fenv.fetestexcept
    libc.src.__support.FPUtil.fenv_impl
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  getenv_and_setenv_test
  SUITE
    libc_fenv_tests
  SRCS
    getenv_and_setenv_test.cpp
  DEPENDS
    libc.src.fenv.fegetenv
    libc.src.fenv.fegetround
    libc.src.fenv.fesetenv
    libc.src.fenv.fesetround
    libc.src.fenv.feclearexcept
    libc.src.fenv.feraiseexcept
    libc.src.fenv.fetestexcept
    libc.src.__support.FPUtil.fenv_impl
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  exception_flags_test
  SUITE
    libc_fenv_tests
  SRCS
    exception_flags_test.cpp
  DEPENDS
    libc.src.fenv.fegetexceptflag
    libc.src.fenv.fesetexceptflag
    libc.src.fenv.fetestexceptflag
    libc.src.__support.FPUtil.fenv_impl
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  feupdateenv_test
  SUITE
    libc_fenv_tests
  SRCS
    feupdateenv_test.cpp
  DEPENDS
    libc.include.signal
    libc.src.fenv.feupdateenv
    libc.src.__support.FPUtil.fenv_impl
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  feclearexcept_test
  SUITE
    libc_fenv_tests
  SRCS
    feclearexcept_test.cpp
  DEPENDS
    libc.src.fenv.feclearexcept
    libc.src.__support.FPUtil.fenv_impl
  LINK_LIBRARIES
    LibcFPTestHelpers
)

add_libc_test(
  feenableexcept_test
  SUITE
    libc_fenv_tests
  SRCS
    feenableexcept_test.cpp
  DEPENDS
    libc.src.fenv.fedisableexcept
    libc.src.fenv.feenableexcept
    libc.src.fenv.fegetexcept
  LINK_LIBRARIES
    LibcFPTestHelpers
)

if (NOT (LLVM_USE_SANITIZER OR (${LIBC_TARGET_OS} STREQUAL "windows")
         OR (${LIBC_TARGET_OS} STREQUAL "darwin")))
  # Sanitizers don't like SIGFPE. So, we will run the
  # tests which raise SIGFPE only in non-sanitizer builds.
  # The tests are also disabled for Windows and MacOS as they fail currently.
  # TODO: Investigate and fix the windows failures and enable them for Windows
  # and MacOS.
  add_fp_unittest(
    enabled_exceptions_test
    UNIT_TEST_ONLY
    SUITE
      libc_fenv_tests
    SRCS
      enabled_exceptions_test.cpp
    DEPENDS
      libc.include.signal
      libc.src.fenv.feclearexcept
      libc.src.fenv.feraiseexcept
      libc.src.fenv.fetestexcept
      libc.src.__support.FPUtil.fenv_impl
    LINK_LIBRARIES
      LibcFPExceptionHelpers
      LibcFPTestHelpers
  )

  add_fp_unittest(
    feholdexcept_test
    UNIT_TEST_ONLY
    SUITE
      libc_fenv_tests
    SRCS
      feholdexcept_test.cpp
    DEPENDS
      libc.hdr.fenv_macros
      libc.src.fenv.feholdexcept
      libc.src.__support.FPUtil.fenv_impl
    LINK_LIBRARIES
      LibcFPExceptionHelpers
      LibcFPTestHelpers
  )
endif()
