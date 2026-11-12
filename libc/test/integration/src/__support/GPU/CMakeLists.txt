add_libc_testsuite(libc-support-gpu-integration-tests PARENT libc-integration-tests)

add_integration_test(
  scan_reduce_test
  SUITE libc-support-gpu-integration-tests
  SRCS
    scan_reduce.cpp
  LOADER_ARGS
    --threads 64
)

add_integration_test(
  shuffle_test
  SUITE libc-support-gpu-integration-tests
  SRCS
    shuffle.cpp
  LOADER_ARGS
    --threads 64
)

add_integration_test(
  match_test
  SUITE libc-support-gpu-integration-tests
  SRCS
    match.cpp
  LOADER_ARGS
    --threads 64
)

add_integration_test(
  fixedbuffer_test
  SUITE
    libc-support-gpu-integration-tests
  SRCS
    fixedbuffer_test.cpp
  DEPENDS
    libc.src.__support.GPU.fixedbuffer
  LOADER_ARGS
    --threads 32
    --blocks 16
)
