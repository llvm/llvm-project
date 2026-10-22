if(LIBC_TARGET_OS_IS_GPU)
  return()
endif()

add_libc_testsuite(libc-unittest-tests)

add_libc_test(
  testfilter_test
  SUITE
    libc-unittest-tests
  SRCS
    testfilter_test.cpp
  DEPENDS
    # TODO(michaelrj): Remove this dependancy. It's only here because all unit
    # tests must have at least one dependancy.
    libc.src.__support.CPP.bit
)
