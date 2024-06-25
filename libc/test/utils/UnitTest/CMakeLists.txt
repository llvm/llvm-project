if(LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_custom_target(libc_unittest_tests)

add_libc_unittest(
  testfilter_test
  SUITE
    libc_unittest_tests
  SRCS
    testfilter_test.cpp
  DEPENDS
    # TODO(michaelrj): Remove this dependancy. It's only here because all unit
    # tests must have at least one dependancy.
    libc.src.__support.CPP.bit
)
