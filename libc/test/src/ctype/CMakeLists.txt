add_libc_testsuite(libc_ctype_unittests)

add_libc_unittest(
  isalnum
  SUITE
    libc_ctype_unittests
  SRCS
    isalnum_test.cpp
  DEPENDS
    libc.src.ctype.isalnum
)

add_libc_unittest(
  isalpha
  SUITE
    libc_ctype_unittests
  SRCS
    isalpha_test.cpp
  DEPENDS
    libc.src.ctype.isalpha
)

add_libc_unittest(
  isdigit
  SUITE
    libc_ctype_unittests
  SRCS
    isdigit_test.cpp
  DEPENDS
    libc.src.ctype.isdigit
)

add_libc_unittest(
  islower
  SUITE
    libc_ctype_unittests
  SRCS
    islower_test.cpp
  DEPENDS
    libc.src.ctype.islower
)

add_libc_unittest(
  isupper
  SUITE
    libc_ctype_unittests
  SRCS
    isupper_test.cpp
  DEPENDS
    libc.src.ctype.isupper
)
