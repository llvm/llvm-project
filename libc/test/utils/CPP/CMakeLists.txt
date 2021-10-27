add_libc_testsuite(libc_cpp_utils_unittests)

add_libc_unittest(
  bitset_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    bitset_test.cpp
  DEPENDS
    libc.src.__support.CPP.standalone_cpp
)

add_libc_unittest(
  stringview_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    stringview_test.cpp
  DEPENDS
    libc.src.__support.CPP.standalone_cpp
)

add_libc_unittest(
  limits_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    limits_test.cpp
  DEPENDS
    libc.src.__support.CPP.standalone_cpp
)

add_libc_unittest(
  arrayref_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    arrayref_test.cpp
  DEPENDS
    libc.src.__support.CPP.standalone_cpp
)
