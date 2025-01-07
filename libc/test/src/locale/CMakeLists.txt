add_custom_target(libc-locale-tests)

add_libc_test(
  locale_test
  SUITE
    libc-locale-tests
  SRCS
    locale_test.cpp
  DEPENDS
    libc.include.locale
    libc.src.locale.newlocale
    libc.src.locale.uselocale
    libc.src.locale.freelocale
)

add_libc_test(
  localeconv_test
  SUITE
    libc-locale-tests
  SRCS
    localeconv_test.cpp
  DEPENDS
    libc.include.locale
    libc.src.locale.localeconv
)
