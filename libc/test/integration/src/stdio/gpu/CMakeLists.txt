add_libc_testsuite(libc-stdio-gpu-integration-tests PARENT libc-stdio-integration-tests)

# Create an output directory for any temporary test files.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testdata)

# These tests are not for correctness testing, but are instead a convenient way
# to generate hermetic binaries for comparitive binary size testing.
add_integration_test(
  printf_test
  SUITE
    libc-stdio-gpu-integration-tests
  SRCS
    printf_test.cpp
  DEPENDS
    libc.src.stdio.fprintf
    libc.src.stdio.fopen
  LOADER_ARGS
    --threads 32
    --blocks 4
)
