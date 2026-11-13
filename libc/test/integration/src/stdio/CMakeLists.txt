add_libc_testsuite(libc-stdio-integration-tests PARENT libc-integration-tests)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()

# These tests are not for correctness testing, but are instead a convenient way
# to generate hermetic binaries for comparitive binary size testing.
# FIXME: This test fails on AMDGPU so we disable it temporarily.
if(NOT LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_integration_test(
    sprintf_size_test
    SUITE
      libc-stdio-integration-tests
    SRCS
      sprintf_size_test.cpp
    DEPENDS
      libc.src.stdio.sprintf
    ARGS
      "%s %c %d"
      "First arg"
      "a"
      "0"
  )
endif()

add_integration_test(
  sprintf_size_test_no_sprintf
  SUITE
    libc-stdio-integration-tests
  SRCS
    sprintf_size_test.cpp
  ARGS
    "%s %c %d"
    "First arg"
    "a"
    "0"
  COMPILE_OPTIONS
    -DINTEGRATION_DISABLE_PRINTF
)
