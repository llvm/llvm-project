add_custom_target(libc-ctype-tests)

add_libc_test(
  isalnum_test
  SUITE
    libc-ctype-tests
  SRCS
    isalnum_test.cpp
  DEPENDS
    libc.src.ctype.isalnum
)

add_libc_test(
  isalpha_test
  SUITE
    libc-ctype-tests
  SRCS
    isalpha_test.cpp
  DEPENDS
    libc.src.ctype.isalpha
)

add_libc_test(
  isascii_test
  SUITE
    libc-ctype-tests
  SRCS
    isascii_test.cpp
  DEPENDS
    libc.src.ctype.isascii
)

add_libc_test(
  isblank_test
  SUITE
    libc-ctype-tests
  SRCS
    isblank_test.cpp
  DEPENDS
    libc.src.ctype.isblank
)

add_libc_test(
  iscntrl_test
  SUITE
    libc-ctype-tests
  SRCS
    iscntrl_test.cpp
  DEPENDS
    libc.src.ctype.iscntrl
)

add_libc_test(
  isdigit_test
  SUITE
    libc-ctype-tests
  SRCS
    isdigit_test.cpp
  DEPENDS
    libc.src.ctype.isdigit
)

add_libc_test(
  isgraph_test
  SUITE
    libc-ctype-tests
  SRCS
    isgraph_test.cpp
  DEPENDS
    libc.src.ctype.isgraph
)

add_libc_test(
  islower_test
  SUITE
    libc-ctype-tests
  SRCS
    islower_test.cpp
  DEPENDS
    libc.src.ctype.islower
)

add_libc_test(
  isprint_test
  SUITE
    libc-ctype-tests
  SRCS
    isprint_test.cpp
  DEPENDS
    libc.src.ctype.isprint
)

add_libc_test(
  ispunct_test
  SUITE
    libc-ctype-tests
  SRCS
    ispunct_test.cpp
  DEPENDS
    libc.src.ctype.ispunct
)

add_libc_test(
  isspace_test
  SUITE
    libc-ctype-tests
  SRCS
    isspace_test.cpp
  DEPENDS
    libc.src.ctype.isspace
)

add_libc_test(
  isupper_test
  SUITE
    libc-ctype-tests
  SRCS
    isupper_test.cpp
  DEPENDS
    libc.src.ctype.isupper
)

add_libc_test(
  isxdigit_test
  SUITE
    libc-ctype-tests
  SRCS
    isxdigit_test.cpp
  DEPENDS
    libc.src.ctype.isxdigit
)

add_libc_test(
  toascii_test
  SUITE
    libc-ctype-tests
  SRCS
    toascii_test.cpp
  DEPENDS
    libc.src.ctype.toascii
)

add_libc_test(
  tolower_test
  SUITE
    libc-ctype-tests
  SRCS
    tolower_test.cpp
  DEPENDS
    libc.src.ctype.tolower
)

add_libc_test(
  toupper_test
  SUITE
    libc-ctype-tests
  SRCS
    toupper_test.cpp
  DEPENDS
    libc.src.ctype.toupper
)
