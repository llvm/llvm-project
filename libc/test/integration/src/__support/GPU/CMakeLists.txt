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

add_integration_test(
  shuffle_test
  SUITE libc-support-gpu-tests
  SRCS
    shuffle.cpp
  LOADER_ARGS
    --threads 64
)

add_integration_test(
  match_test
  SUITE libc-support-gpu-tests
  SRCS
    match.cpp
  LOADER_ARGS
    --threads 64
)

add_libc_test(
  fixedstack_test
  SUITE
    libc-support-gpu-tests
  SRCS
    fixedstack_test.cpp
  DEPENDS
    libc.src.__support.GPU.fixedstack
  LOADER_ARGS
    --threads 32
    --blocks 16
)
