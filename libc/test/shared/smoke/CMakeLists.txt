add_custom_target(libc-shared-math-smoke-tests)
# add_dependencies(libc-shared-unittests libc-shared-smoke-tests)

add_fp_unittest(
  acos_test
  SUITE
    libc-shared-math-smoke-tests
  SRCS
    acos_test.cpp
  DEPENDS
    libc.src.errno.errno
    libc.src.__support.math.acos
)
