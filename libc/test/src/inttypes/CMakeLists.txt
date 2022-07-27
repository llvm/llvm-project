add_libc_testsuite(libc_inttypes_unittests)

add_libc_unittest(
  strtoimax_test
  SUITE
    libc_inttypes_unittests
  SRCS
    strtoimax_test.cpp
  DEPENDS
    libc.src.inttypes.strtoimax
)

add_libc_unittest(
  strtoumax_test
  SUITE
    libc_inttypes_unittests
  SRCS
    strtoumax_test.cpp
  DEPENDS
    libc.src.inttypes.strtoumax
)

add_libc_unittest(
  imaxdiv_test
  SUITE
    libc_inttypes_unittests
  SRCS
    imaxdiv_test.cpp
  HDRS
    ../stdlib/DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.inttypes.imaxdiv
)

add_libc_unittest(
  imaxabs_test
  SUITE
    libc_inttypes_unittests
  SRCS
    imaxabs_test.cpp
  HDRS
    ../stdlib/DivTest.h
  DEPENDS
    libc.include.stdlib
    libc.src.inttypes.imaxabs
)

