add_libc_testsuite(libc_ctype_unittests)

add_libc_unittest(
  isalnum_test
  SUITE
    libc_ctype_unittests
  SRCS
    isalnum_test.cpp
  DEPENDS
    libc.src.ctype.isalnum
)

add_libc_unittest(
  isalpha_test
  SUITE
    libc_ctype_unittests
  SRCS
    isalpha_test.cpp
  DEPENDS
    libc.src.ctype.isalpha
)

add_libc_unittest(
  isascii_test
  SUITE
    libc_ctype_unittests
  SRCS
    isascii_test.cpp
  DEPENDS
    libc.src.ctype.isascii
)

add_libc_unittest(
  isblank_test
  SUITE
    libc_ctype_unittests
  SRCS
    isblank_test.cpp
  DEPENDS
    libc.src.ctype.isblank
)

add_libc_unittest(
  iscntrl_test
  SUITE
    libc_ctype_unittests
  SRCS
    iscntrl_test.cpp
  DEPENDS
    libc.src.ctype.iscntrl
)

add_libc_unittest(
  isdigit_test
  SUITE
    libc_ctype_unittests
  SRCS
    isdigit_test.cpp
  DEPENDS
    libc.src.ctype.isdigit
)

add_libc_unittest(
  isgraph_test
  SUITE
    libc_ctype_unittests
  SRCS
    isgraph_test.cpp
  DEPENDS
    libc.src.ctype.isgraph
)

add_libc_unittest(
  islower_test
  SUITE
    libc_ctype_unittests
  SRCS
    islower_test.cpp
  DEPENDS
    libc.src.ctype.islower
)

add_libc_unittest(
  isprint_test
  SUITE
    libc_ctype_unittests
  SRCS
    isprint_test.cpp
  DEPENDS
    libc.src.ctype.isprint
)

add_libc_unittest(
  ispunct_test
  SUITE
    libc_ctype_unittests
  SRCS
    ispunct_test.cpp
  DEPENDS
    libc.src.ctype.ispunct
)

add_libc_unittest(
  isspace_test
  SUITE
    libc_ctype_unittests
  SRCS
    isspace_test.cpp
  DEPENDS
    libc.src.ctype.isspace
)

add_libc_unittest(
  isupper_test
  SUITE
    libc_ctype_unittests
  SRCS
    isupper_test.cpp
  DEPENDS
    libc.src.ctype.isupper
)

add_libc_unittest(
  isxdigit_test
  SUITE
    libc_ctype_unittests
  SRCS
    isxdigit_test.cpp
  DEPENDS
    libc.src.ctype.isxdigit
)

add_libc_unittest(
  toascii_test
  SUITE
    libc_ctype_unittests
  SRCS
    toascii_test.cpp
  DEPENDS
    libc.src.ctype.toascii
)

add_libc_unittest(
  tolower_test
  SUITE
    libc_ctype_unittests
  SRCS
    tolower_test.cpp
  DEPENDS
    libc.src.ctype.tolower
)

add_libc_unittest(
  toupper_test
  SUITE
    libc_ctype_unittests
  SRCS
    toupper_test.cpp
  DEPENDS
    libc.src.ctype.toupper
)
