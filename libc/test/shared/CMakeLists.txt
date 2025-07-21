add_custom_target(libc-shared-tests)

add_fp_unittest(
  shared_math_test
  SUITE
    libc-shared-tests
  SRCS
    shared_math_test.cpp
)
