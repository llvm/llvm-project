add_custom_target(support-gpu-integration-tests)
add_dependencies(libc-integration-tests support-gpu-integration-tests)

add_integration_test(
  support_fixed_stack_test
  SUITE support-gpu-integration-tests
  SRCS
    fixed_stack_test.cpp
  DEPENDS
    libc.src.__support.GPU.utils
    libc.src.__support.fixedstack
  LOADER_ARGS
    --blocks-x 2
    --blocks-y 2
    --blocks-z 2
    --threads-x 4
    --threads-y 4
    --threads-z 4
)
