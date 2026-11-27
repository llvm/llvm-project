add_libc_testsuite(libc-link-tests)

add_libc_test(
  dl_iterate_phdr_test
  SUITE
    libc-link-tests
  SRCS
    dl_iterate_phdr_test.cpp
  DEPENDS
    libc.src.link.dl_iterate_phdr
)
