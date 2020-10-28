add_libc_testsuite(libc_cpp_utils_unittests)

add_libc_unittest(
  bitset_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    bitset_test.cpp
  DEPENDS
    libc.utils.CPP.standalone_cpp
)
