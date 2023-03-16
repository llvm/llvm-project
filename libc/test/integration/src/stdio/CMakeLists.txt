add_custom_target(stdio-integration-tests)
add_dependencies(libc-integration-tests stdio-integration-tests)

# These tests are not for correctness testing, but are instead a convenient way
# to generate hermetic binaries for comparitive binary size testing.
add_integration_test(
  sprintf_size_test
  SUITE
    stdio-integration-tests
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

add_integration_test(
  sprintf_size_test_no_sprintf
  SUITE
    stdio-integration-tests
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

