add_custom_target(libc-shared-math-smoke-tests)

add_fp_unittest(
  shared_math_test
  SUITE
    libc-shared-math-smoke-tests
  SRCS
    shared_math_test.cpp
)
