add_libc_testsuite(libc_unittest_tests)

add_libc_unittest(
  testfilter_test
  SUITE
    libc_unittest_tests
  SRCS
    testfilter_test.cpp
  DEPENDS
    libc.utils.CPP.standalone_cpp
)
