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
    --blocks 16
    --threads 1
)

# Constructors are currently only supported on AMDGPU.
if(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU)
  add_integration_test(
    init_fini_array_test
    SUITE libc-startup-tests
    SRCS
      init_fini_array_test.cpp
  )
endif()
