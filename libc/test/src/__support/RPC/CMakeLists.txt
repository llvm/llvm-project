add_libc_testsuite(libc-rpc-tests PARENT libc-support-tests)

add_libc_test(
  rpc_smoke_test
  SUITE
    libc-rpc-tests
  SRCS
    rpc_smoke_test.cpp
  DEPENDS
    libc.src.__support.RPC.rpc
)

add_libc_test(
  rpc_doorbell_test
  SUITE
    libc-rpc-tests
  SRCS
    rpc_doorbell_test.cpp
  DEPENDS
    libc.src.__support.RPC.rpc
)
