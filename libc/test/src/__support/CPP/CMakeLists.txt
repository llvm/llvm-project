add_libc_testsuite(libc_cpp_utils_unittests)

add_libc_unittest(
  bitset_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    bitset_test.cpp
  DEPENDS
    libc.src.__support.CPP.bitset
)

add_libc_unittest(
  stringview_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    stringview_test.cpp
  DEPENDS
    libc.src.__support.CPP.string_view
)

add_libc_unittest(
  limits_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    limits_test.cpp
  DEPENDS
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.uint
)

add_libc_unittest(
  arrayref_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    arrayref_test.cpp
  DEPENDS
    libc.src.__support.CPP.array_ref
)

add_libc_unittest(
  vector_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    vector_test.cpp
  DEPENDS
    libc.src.__support.CPP.vector
)

add_libc_unittest(
  int_seq_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    integer_sequence_test.cpp
  DEPENDS
    libc.src.__support.CPP.utility
)

add_libc_unittest(
  atomic_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    atomic_test.cpp
  DEPENDS
    libc.src.__support.CPP.atomic
)

add_libc_unittest(
  blockstore_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    blockstore_test.cpp
  DEPENDS
    libc.src.__support.CPP.blockstore
)

add_libc_unittest(
  stringstream_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    stringstream_test.cpp
  DEPENDS
    libc.src.__support.CPP.array_ref
    libc.src.__support.CPP.stringstream
)

add_libc_unittest(
  optional_test
  SUITE
    libc_cpp_utils_unittests
  SRCS
    optional_test.cpp
  DEPENDS
  libc.src.__support.CPP.optional
)
