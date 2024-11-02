add_custom_target(libc-support-gpu-tests)
add_dependencies(libc-integration-tests libc-support-gpu-tests)

add_integration_test(
  scan_reduce_test
  SUITE libc-support-gpu-tests
  SRCS
    scan_reduce.cpp
  LOADER_ARGS
    --threads 64
)
