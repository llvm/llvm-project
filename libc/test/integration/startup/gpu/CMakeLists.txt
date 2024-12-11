add_custom_target(libc-startup-tests)
add_dependencies(libc-integration-tests libc-startup-tests)

add_integration_test(
  startup_args_test
  SUITE libc-startup-tests
  SRCS
    args_test.cpp
  ARGS
    1 2 3
  ENV
    FRANCE=Paris
    GERMANY=Berlin
)

add_integration_test(
  startup_rpc_test
  SUITE libc-startup-tests
  SRCS
    rpc_test.cpp
  DEPENDS
    libc.src.__support.RPC.rpc_client
    libc.src.__support.GPU.utils
  LOADER_ARGS
    --blocks-x 2
    --blocks-y 2
    --blocks-z 2
    --threads-x 4
    --threads-y 4
    --threads-z 4
)

add_integration_test(
  init_fini_array_test
  SUITE libc-startup-tests
  SRCS
    init_fini_array_test.cpp
)

add_integration_test(
  startup_rpc_interface_test
  SUITE libc-startup-tests
  SRCS
   rpc_interface_test.cpp
)

add_integration_test(
  startup_rpc_stream_test
  SUITE libc-startup-tests
  SRCS
   rpc_stream_test.cpp
  LOADER_ARGS
   --threads 32
   --blocks 8
)
