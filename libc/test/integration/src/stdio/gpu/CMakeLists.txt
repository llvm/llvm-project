add_custom_target(stdio-gpu-integration-tests)
add_dependencies(libc-integration-tests stdio-gpu-integration-tests)

# Create an output directory for any temporary test files.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testdata)

# These tests are not for correctness testing, but are instead a convenient way
# to generate hermetic binaries for comparitive binary size testing.
add_integration_test(
  printf_test
  SUITE
    stdio-gpu-integration-tests
  SRCS
    printf.cpp
  DEPENDS
    libc.src.gpu.rpc_fprintf
    libc.src.stdio.fopen
  LOADER_ARGS
    --threads 32
    --blocks 4
)
