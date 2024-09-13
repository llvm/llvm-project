add_custom_target(libc-support-tests)

if(NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_libc_test(
    block_test
    SUITE
      libc-support-tests
    SRCS
      block_test.cpp
    DEPENDS
      libc.src.__support.CPP.array
      libc.src.__support.CPP.bit
      libc.src.__support.CPP.span
      libc.src.__support.block
      libc.src.string.memcpy
  )

  add_libc_test(
    freelist_test
    SUITE
      libc-support-tests
    SRCS
      freelist_test.cpp
    DEPENDS
      libc.src.__support.CPP.array
      libc.src.__support.CPP.span
      libc.src.__support.freelist
  )
endif()

if(LLVM_LIBC_FULL_BUILD)
  add_libc_test(
    freelist_heap_test
    SUITE
      libc-support-tests
    SRCS
      fake_heap.s
      freelist_heap_test.cpp
      freelist_malloc_test.cpp
    DEPENDS
      libc.src.__support.CPP.span
      libc.src.__support.freelist_heap
      libc.src.stdlib.freelist_malloc
      libc.src.string.memcmp
      libc.src.string.memcpy
  )
endif()

add_libc_test(
  blockstore_test
  SUITE
    libc-support-tests
  SRCS
    blockstore_test.cpp
  DEPENDS
    libc.src.__support.blockstore
)

add_libc_test(
  endian_test
  SUITE
    libc-support-tests
  SRCS
    endian_test.cpp
  DEPENDS
    libc.src.__support.common
)

add_libc_test(
  math_extras_test
  SUITE
    libc-support-tests
  SRCS
    math_extras_test.cpp
  DEPENDS
    libc.src.__support.integer_literals
    libc.src.__support.math_extras
    libc.src.__support.uint128
)

add_libc_test(
  high_precision_decimal_test
  SUITE
    libc-support-tests
  SRCS
  high_precision_decimal_test.cpp
  DEPENDS
    libc.src.__support.high_precision_decimal
    libc.src.__support.uint128
)

add_libc_test(
  str_to_float_test
  SUITE
    libc-support-tests
  SRCS
    str_to_float_test.cpp
    str_to_double_test.cpp
    str_to_long_double_test.cpp
  DEPENDS
    libc.src.__support.integer_literals
    libc.src.__support.str_to_float
    libc.src.__support.uint128
    libc.src.errno.errno
)


add_libc_test(
  str_to_integer_test
  SUITE
    libc-support-tests
  SRCS
    str_to_integer_test.cpp
  DEPENDS
    libc.src.__support.integer_literals
    libc.src.__support.str_to_integer
    libc.src.errno.errno
)

add_libc_test(
  integer_to_string_test
  SUITE
    libc-support-tests
  SRCS
    integer_to_string_test.cpp
  DEPENDS
    libc.src.__support.big_int
    libc.src.__support.CPP.limits
    libc.src.__support.CPP.string_view
    libc.src.__support.integer_literals
    libc.src.__support.integer_to_string
    libc.src.__support.uint128
)

add_libc_test(
  arg_list_test
  SUITE
    libc-support-tests
  SRCS
    arg_list_test.cpp
  DEPENDS
    libc.src.__support.arg_list
    libc.src.__support.macros.properties.os
)

# TODO: clang-cl generates calls into runtime library functions to
# handle 128-bit integer arithmetics and conversions which are not yet
# available on Windows. Re-enable 128-bit integer support on Windows once
# these functions are ready.
if(NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX AND NOT LIBC_TARGET_OS_IS_WINDOWS)
  add_libc_test(
    big_int_test
    SUITE
      libc-support-tests
    SRCS
      big_int_test.cpp
    DEPENDS
      libc.src.__support.big_int
      libc.src.__support.CPP.optional
      libc.src.__support.macros.properties.types
  )
endif()

add_libc_test(
  integer_literals_test
  SUITE
    libc-support-tests
  SRCS
    integer_literals_test.cpp
  DEPENDS
    libc.src.__support.CPP.optional
    libc.src.__support.integer_literals
    libc.src.__support.macros.properties.types
)

add_libc_test(
  fixedvector_test
  SUITE
    libc-support-tests
  SRCS
    fixedvector_test.cpp
  DEPENDS
    libc.src.__support.CPP.array
    libc.src.__support.fixedvector
)

add_libc_test(
  char_vector_test
  SUITE
    libc-support-tests
  SRCS
    char_vector_test.cpp
  DEPENDS
    libc.src.__support.char_vector
)

add_libc_test(
  hash_test
  SUITE
    libc-support-tests
  SRCS
    hash_test.cpp
  DEPENDS
    libc.src.__support.hash
    libc.src.__support.CPP.new
    libc.src.stdlib.rand
    libc.src.stdlib.srand
    libc.src.string.memset
  COMPILE_OPTIONS
    -O3
  UNIT_TEST_ONLY
    # Aligned Allocation is not supported in hermetic builds.
)

add_libc_test(
  memory_size_test
  SUITE
    libc-support-tests
  SRCS
    memory_size_test.cpp
  DEPENDS
    libc.src.__support.memory_size
)

# FIXME: We shouldn't have regular executables created because we could be
#        cross-compiling the tests and running through an emulator.
if(NOT LIBC_TARGET_OS_IS_GPU)
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
                     COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} $<TARGET_FILE:libc_str_to_float_comparison_test> ${float_test_file}
                     DEPENDS ${float_test_file}
                     COMMENT "Test the strtof and strtod implementations against precomputed results."
                     VERBATIM)
endif()

add_subdirectory(CPP)
add_subdirectory(File)
add_subdirectory(RPC)
add_subdirectory(OSUtil)
add_subdirectory(FPUtil)
add_subdirectory(fixed_point)
add_subdirectory(HashTable)
add_subdirectory(time)
add_subdirectory(threads)
