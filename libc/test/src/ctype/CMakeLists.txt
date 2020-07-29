add_libc_testsuite(libc_ctype_unittests)

add_libc_unittest(
  isalpha
  SUITE
    libc_ctype_unittests
  SRCS
    isalpha_test.cpp
  DEPENDS
    libc.src.ctype.isalpha
)
