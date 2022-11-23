add_libc_testsuite(libc_support_unittests)

add_libc_unittest(
  blockstore_test
  SUITE
    libc_support_unittests
  SRCS
    blockstore_test.cpp
  DEPENDS
    libc.src.__support.blockstore
)

add_libc_unittest(
  endian_test
  SUITE
    libc_support_unittests
  SRCS
    endian_test.cpp
  DEPENDS
    libc.src.__support.common
)

add_libc_unittest(
  high_precision_decimal_test
  SUITE
    libc_support_unittests
  SRCS
  high_precision_decimal_test.cpp
  DEPENDS
    libc.src.__support.high_precision_decimal
    libc.src.__support.uint128
)

add_libc_unittest(
  str_to_float_test
  SUITE
    libc_support_unittests
  SRCS
    str_to_float_test.cpp
  DEPENDS
    libc.src.__support.str_to_float
    libc.src.__support.uint128
)

add_libc_unittest(
  integer_to_string_test
  SUITE
    libc_support_unittests
  SRCS
    integer_to_string_test.cpp
  DEPENDS
    libc.src.__support.integer_to_string
    libc.src.__support.CPP.string_view
)

add_libc_unittest(
  arg_list_test
  SUITE
    libc_support_unittests
  SRCS
    arg_list_test.cpp
  DEPENDS
    libc.src.__support.arg_list
)

add_libc_unittest(
  uint_test
  SUITE
    libc_support_unittests
  SRCS
    uint_test.cpp
  DEPENDS
    libc.src.__support.uint
    libc.src.__support.CPP.optional
)

add_libc_unittest(
  fixedvector_test
  SUITE
    libc_support_unittests
  SRCS
    fixedvector_test.cpp
  DEPENDS
    libc.src.__support.fixedvector
)

add_executable(
  libc_str_to_float_comparison_test
  str_to_float_comparison_test.cpp
)

target_link_libraries(libc_str_to_float_comparison_test
  PRIVATE
    "${LIBC_TARGET}"
)

add_executable(
  libc_system_str_to_float_comparison_test
  str_to_float_comparison_test.cpp
)

set(float_test_file ${CMAKE_CURRENT_SOURCE_DIR}/str_to_float_comparison_data.txt)

add_custom_command(TARGET libc_str_to_float_comparison_test
                   POST_BUILD
                   COMMAND $<TARGET_FILE:libc_str_to_float_comparison_test> ${float_test_file}
                   DEPENDS ${float_test_file}
                   COMMENT "Test the strtof and strtod implementations against precomputed results." 
                   VERBATIM)

add_subdirectory(CPP)
add_subdirectory(File)
add_subdirectory(OSUtil)
