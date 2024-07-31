add_custom_target(libc_inttypes_tests)

add_libc_test(
  strtoimax_test
  SUITE
    libc_inttypes_tests
  SRCS
    strtoimax_test.cpp
  DEPENDS
    libc.src.inttypes.strtoimax
    libc.test.src.stdlib.strtol_test_support
)

add_libc_test(
  strtoumax_test
  SUITE
    libc_inttypes_tests
  SRCS
    strtoumax_test.cpp
  DEPENDS
    libc.src.inttypes.strtoumax
    libc.test.src.stdlib.strtol_test_support
)

add_libc_test(
  imaxdiv_test
  SUITE
    libc_inttypes_tests
  SRCS
    imaxdiv_test.cpp
  HDRS
    ../stdlib/DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.inttypes.imaxdiv
)

add_libc_test(
  imaxabs_test
  SUITE
    libc_inttypes_tests
  SRCS
    imaxabs_test.cpp
  HDRS
    ../stdlib/DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.inttypes.imaxabs
)

