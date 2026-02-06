add_custom_target(libc-cpp-utils-tests)

add_libc_test(
  algorithm_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    algorithm_test.cpp
  DEPENDS
    libc.src.__support.CPP.algorithm
  )

add_libc_test(
  array_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    array_test.cpp
  DEPENDS
    libc.src.__support.CPP.array
  )

add_libc_test(
  bit_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    bit_test.cpp
  DEPENDS
    libc.hdr.stdint_proxy
    libc.src.__support.big_int
    libc.src.__support.CPP.bit
    libc.src.__support.macros.properties.types
)

add_libc_test(
  bitset_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    bitset_test.cpp
  DEPENDS
    libc.src.__support.CPP.bitset
)

add_libc_test(
  cstddef_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    cstddef_test.cpp
  DEPENDS
    libc.src.__support.CPP.cstddef
)

add_libc_test(
  stringview_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    stringview_test.cpp
  DEPENDS
    libc.src.__support.CPP.string_view
)

add_libc_test(
  limits_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    limits_test.cpp
  DEPENDS
    libc.src.__support.big_int
    libc.src.__support.CPP.limits
    libc.src.__support.macros.properties.types
)

add_libc_test(
  mutex_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    mutex_test.cpp
  DEPENDS
    libc.src.__support.CPP.mutex
)

add_libc_test(
  int_seq_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    integer_sequence_test.cpp
  DEPENDS
    libc.src.__support.CPP.utility
)


# This test fails with invalid address space operations on sm_60
if(NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_libc_test(
    atomic_test
    SUITE
      libc-cpp-utils-tests
    SRCS
      atomic_test.cpp
    DEPENDS
      libc.src.__support.CPP.atomic
  )
endif()

add_libc_test(
  stringstream_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    stringstream_test.cpp
  DEPENDS
    libc.src.__support.CPP.span
    libc.src.__support.CPP.stringstream
)

add_libc_test(
  optional_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    optional_test.cpp
  DEPENDS
    libc.src.__support.CPP.optional
)

add_libc_test(
  tuple_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    tuple_test.cpp
  DEPENDS
    libc.src.__support.CPP.tuple
)

add_libc_test(
  span_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    span_test.cpp
  DEPENDS
    libc.src.__support.CPP.span
)

add_libc_test(
  string_test
  SUITE
    libc-cpp-utils-tests
  SRCS
    string_test.cpp
  DEPENDS
  libc.src.__support.CPP.string
  libc.src.__support.CPP.string_view
)

add_libc_test(
  type_traits_test
  SUITE
    libc-cpp-utils-tests
  SRCS
  type_traits_test.cpp
  DEPENDS
  libc.src.__support.CPP.type_traits
)

if(LIBC_COMPILER_HAS_EXT_VECTOR_TYPE)
  add_libc_test(
    simd_test
    SUITE
      libc-cpp-utils-tests
    SRCS
      simd_test.cpp
    DEPENDS
      libc.src.__support.CPP.simd
  )
endif()
