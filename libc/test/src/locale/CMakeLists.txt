add_custom_target(libc-locale-tests)

add_libc_test(
  locale_test
  SUITE
    libc-locale-tests
  SRCS
    locale_test.cpp
  DEPENDS
    libc.hdr.errno_macros
    libc.hdr.locale_macros
    libc.src.__support.CPP.scope
    libc.src.locale.freelocale
    libc.src.locale.newlocale
    libc.src.locale.uselocale
    libc.test.UnitTest.ErrnoCheckingTest
)

add_libc_test(
  localeconv_test
  SUITE
    libc-locale-tests
  SRCS
    localeconv_test.cpp
  DEPENDS
    libc.hdr.locale_macros
    libc.src.locale.localeconv
)
