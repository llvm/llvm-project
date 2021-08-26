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
