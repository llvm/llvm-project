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
  isblank
  SUITE
    libc_ctype_unittests
  SRCS
    isblank_test.cpp
  DEPENDS
    libc.src.ctype.isblank
)

add_libc_unittest(
  iscntrl
  SUITE
    libc_ctype_unittests
  SRCS
    iscntrl_test.cpp
  DEPENDS
    libc.src.ctype.iscntrl
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
  isgraph
  SUITE
    libc_ctype_unittests
  SRCS
    isgraph_test.cpp
  DEPENDS
    libc.src.ctype.isgraph
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
  isprint
  SUITE
    libc_ctype_unittests
  SRCS
    isprint_test.cpp
  DEPENDS
    libc.src.ctype.isprint
)

add_libc_unittest(
  ispunct
  SUITE
    libc_ctype_unittests
  SRCS
    ispunct_test.cpp
  DEPENDS
    libc.src.ctype.ispunct
)

add_libc_unittest(
  isspace
  SUITE
    libc_ctype_unittests
  SRCS
    isspace_test.cpp
  DEPENDS
    libc.src.ctype.isspace
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

add_libc_unittest(
  isxdigit
  SUITE
    libc_ctype_unittests
  SRCS
    isxdigit_test.cpp
  DEPENDS
    libc.src.ctype.isxdigit
)

add_libc_unittest(
  tolower
  SUITE
    libc_ctype_unittests
  SRCS
    tolower_test.cpp
  DEPENDS
    libc.src.ctype.tolower
)

add_libc_unittest(
  toupper
  SUITE
    libc_ctype_unittests
  SRCS
    toupper_test.cpp
  DEPENDS
    libc.src.ctype.toupper
)
