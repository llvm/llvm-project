add_libc_testsuite(libc-langinfo-tests)

add_libc_test(
  nl_langinfo_test
  SUITE
    libc-langinfo-tests
  SRCS
    nl_langinfo_test.cpp
  DEPENDS
    libc.hdr.langinfo_macros
    libc.hdr.locale_macros
    libc.hdr.types.locale_t
    libc.src.langinfo.nl_langinfo
    libc.src.langinfo.nl_langinfo_l
    libc.src.locale.locale
    libc.test.UnitTest.ErrnoCheckingTest
)
